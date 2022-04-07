"""
Below is an implementation of class-aware resampling,
modified from Tu et al. (https://arxiv.org/abs/2007.09654).
"""
# coding: utf-8
import numpy as np, random, torch
from torch.utils.data import Sampler

from utils import TAGS


class RandomCycleIter:
    def __init__(self, data_list, test_mode=False):
        self.data_list = list(data_list)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
        return self.data_list[self.i]


def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    i, j = 0, 0
    while i < n:
        # yield next(data_iter_list[next(cls_iter)])
        j = 0 if j >= num_samples_cls else j
        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]] * num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        i += 1
        j += 1


class ClassAwareSampler(Sampler):
    def __init__(self, data_source, num_samples_cls=3, reduce=4):
        random.seed(0)
        torch.manual_seed(0)
        self.epoch = 0
        self.num_classes = len(TAGS)
        self.class_counts = data_source.get_class_weights()

        self.class_iter = RandomCycleIter(range(self.num_classes))
        self.cls_data_list = data_source.build_cls_data_list()
        self.data_iter_list = [RandomCycleIter(x) for x in self.cls_data_list]  # repeated
        self.num_samples = int(max(self.class_counts) * self.num_classes / reduce)
        self.num_samples_cls = num_samples_cls

        print(
            "Class-aware Sampler Built! Class number: {}, reduce {}".format(
                self.num_classes, reduce
            )
        )

    def __iter__(self):
        return class_aware_sample_generator(
            self.class_iter, self.data_iter_list, self.num_samples, self.num_samples_cls
        )

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_sample_per_class(self):
        sample_per_cls = np.asarray([len(x) for x in self.gt_labels])
        condition_prob = [
            np.sum(np.asarray(cls_labels), axis=0) / len(cls_labels)
            for cls_labels in self.gt_labels
        ]
        sum_prob = np.sum(condition_prob, axis=0)
        need_sample = sample_per_cls / sum_prob
