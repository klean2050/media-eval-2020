# coding: utf-8
import os, tqdm, argparse
import numpy as np, tqdm
import torch, torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from sklearn.preprocessing import LabelBinarizer

from model import ShortChunkCNN_Res
from data_loader import read_file
from utils import TAGS


class Predict(object):
    def __init__(self, config):
        self.model_load_path = config.model_load_path
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_val_split = config.use_val_split
        self.save_predictions = config.save_predictions
        self.save_path = config.save_path
        self.initialize()

    def initialize(self):
        self.input_length = 73600
        self.model = ShortChunkCNN_Res()
        self.model.to(self.device)

        state = torch.load(self.model_load_path)
        if "spec.mel_scale.fb" in state.keys():
            self.model.spec.mel_scale.fb = state["spec.mel_scale.fb"]
        self.model.load_state_dict(state)

        test_on = "valid" if self.use_val_split else "test"
        test_file = os.path.join(config.splits_path, f"jamendo_moodtheme-{test_on}.tsv")

        self.file_dict = read_file(test_file)
        self.test_list = list(self.file_dict.keys())
        self.mlb = LabelBinarizer().fit(TAGS)

    def get_tensor(self, fn):
        filename = self.file_dict[fn]["path"]
        npy_path = os.path.join(self.data_path, filename)
        raw = np.load(npy_path, mmap_mode="r")
        hop = (len(raw) - self.input_length) // self.batch_size  # split chunk
        return [
            torch.Tensor(raw[i * hop : i * hop + self.input_length]).unsqueeze(0)
            for i in range(self.batch_size)
        ]

    def test(self):
        self.model = self.model.eval()
        est_array = []
        gt_array = []
        losses = []
        reconst_loss = nn.BCELoss()
        for line in tqdm.tqdm(self.test_list):

            x = self.get_tensor(line)
            ground_truth = np.sum(
                self.mlb.transform(self.file_dict[line]["tags"]),
                axis=0,
            )
            x = Variable(x.to(self.device))
            y = torch.tensor([ground_truth.astype("float32") for _ in range(self.batch_size)])
            y = y.to(self.device)

            out = self.model(x)
            out = torch.sigmoid(out)
            loss = reconst_loss(out, y)
            losses.append(float(loss.data))
            out = out.detach().cpu()

            estimated = np.array(out).mean(axis=0)
            est_array.append(estimated)
            gt_array.append(ground_truth)

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        roc_auc = metrics.roc_auc_score(gt_array, est_array, average="macro")
        pr_auc = metrics.average_precision_score(gt_array, est_array, average="macro")

        if self.save_predictions:
            os.makedirs(self.save_path, exist_ok=True)
            np.save(os.path.join(self.save_path, "predictions.npy"), est_array)
            np.save(os.path.join(self.save_path, "ground_truth.npy"), gt_array)

        print("loss: %.4f" % np.mean(losses))
        print("roc_auc: %.4f" % roc_auc)
        print("pr_auc: %.4f" % pr_auc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_load_path", type=str, default="data/pretrained_models/")
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--splits_path", type=str, default="./splits/")
    parser.add_argument("--use_val_split", type=int, default=0)
    parser.add_argument("--save_predictions", type=int, default=0)
    parser.add_argument("--save_path", type=str, default=".")

    config = parser.parse_args()
    p = Predict(config)
    p.test()
