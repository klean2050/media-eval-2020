"""
The training set is composed of songs from
the Jamendo dataset (https://github.com/MTG/mtg-jamendo-dataset), 
the music4all dataset (https://sites.google.com/view/contact4music4all), 
and the Million Song Dataset (http://millionsongdataset.com/).

Both the validation and test sets are from the following Jamendo split:
https://github.com/MTG/mtg-jamendo-dataset/tree/master/data/splits/split-0
"""
# coding: utf-8
import os
import csv
import numpy as np
from torch.utils import data
from sklearn.preprocessing import LabelBinarizer

from utils import TAGS
from class_aware_res import ClassAwareSampler


"""
Below is a function to parse a file that contains train, val, or test splits.
This function assumes that the input tsv file is formatted as follows:

1st column: unique song identifier
2nd column: relative path to the given song, without the file extension
Remaining columns: Tags for the given song, separated by tabs
"""


def read_file(tsv_file):
    tracks = {}
    split = (
        "validation"
        if "validation" in tsv_file
        else "test"
        if "test" in tsv_file
        else "train"
    )

    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter="\t")
        next(reader, None)  # skip header
        for row in reader:
            tracks[row[0]] = {
                "path": os.path.join("jamendo", split, row[1] + ".npy"),
                "tags": row[2:],
            }
    return tracks


class AudioFolder(data.Dataset):
    def __init__(self, root, split, sampling_type, input_length=None, TSV_PATH="."):
        self.root = root
        self.split = split
        self.input_length = input_length
        self.get_songlist(TSV_PATH)
        self.sampling_type = sampling_type
        assert self.split in [
            "TRAIN",
            "VALID",
            "TEST",
        ], "Split should be one of [TRAIN, VALID, TEST]"

    def __getitem__(self, index):
        npy, tag_binary = self.get_npy(index)
        return npy.astype("float32"), tag_binary.astype("float32")

    def get_songlist(self, TSV_PATH):
        self.mlb = LabelBinarizer().fit(TAGS)
        if self.split == "TRAIN":
            train_file = os.path.join(TSV_PATH, "jamendo_moodtheme-train.tsv")
            self.file_dict = read_file(train_file)
        elif self.split == "VALID":
            val_file = os.path.join(TSV_PATH, "jamendo_moodtheme-validation.tsv")
            self.file_dict = read_file(val_file)
        else:
            test_file = os.path.join(TSV_PATH, "jamendo_moodtheme-test.tsv")
            self.file_dict = read_file(test_file)
        self.fl = list(self.file_dict.keys())

    def get_npy(self, index):
        """
        Load an instance from given npy by randomly
        selecting a segment of length specified in main.py.
        """
        if self.sampling_type == "standard":
            index = self.fl[index]

        filename = self.file_dict[index]["path"]
        npy_path = os.path.join(self.root, filename)
        npy = np.load(npy_path, mmap_mode="r")

        ridx = np.floor(np.random.random(1) * (len(npy) - self.input_length))
        npy = np.array(npy[int(ridx) : int(ridx) + self.input_length])
        tag_binary = np.sum(self.mlb.transform(self.file_dict[index]["tags"]), axis=0)
        return npy, tag_binary

    def __len__(self):
        return len(self.fl)

    def get_gt_labels(self):
        all_labels = []
        for key in self.file_dict.keys():
            tag_binary = np.sum(self.mlb.transform(self.file_dict[key]["tags"]), axis=0)
            all_labels.append(tag_binary)
        return np.array(all_labels)

    def get_class_weights(self):
        return np.sum(self.get_gt_labels(), axis=0)

    def build_cls_data_list(self):
        cls_data_list = []
        for tag in TAGS:
            tag_list = [key for key in self.fl if tag in self.file_dict[key]["tags"]]
            cls_data_list.append(tag_list)
        return cls_data_list


def get_audio_loader(
    root,
    batch_size,
    path_to_tsv,
    sampling_type,
    split="TRAIN",
    num_workers=0,
    input_length=None,
):

    dataset = AudioFolder(
        root,
        split=split,
        sampling_type=sampling_type,
        input_length=input_length,
        TSV_PATH=path_to_tsv,
    )
    class_weights = dataset.get_class_weights()

    if sampling_type == "class_aware":
        sampler = ClassAwareSampler(data_source=dataset, reduce=4)
        data_loader = data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
    else:
        data_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
        )

    return data_loader, class_weights
