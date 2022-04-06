"""
Below is an implementation of class-aware resampling,
modified from Tu et al. (https://arxiv.org/abs/2007.09654).
"""

# coding: utf-8
import numpy as np
import torch
import random
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
    i = 0
    j = 0
    while i < n:

        # yield next(data_iter_list[next(cls_iter)])

        if j >= num_samples_cls:
            j = 0
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
        num_classes = len(TAGS)
        self.class_counts = data_source.get_class_weights()

        self.epoch = 0

        self.class_iter = RandomCycleIter(range(num_classes))

        self.cls_data_list = data_source.build_cls_data_list()

        self.num_classes = len(TAGS)
        self.data_iter_list = [
            RandomCycleIter(x) for x in self.cls_data_list
        ]  # repeated
        self.num_samples = int(max(self.class_counts) * self.num_classes / reduce)
        self.num_samples_cls = num_samples_cls
        print(
            ">>> Class Aware Sampler Built! Class number: {}, reduce {}".format(
                num_classes, reduce
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
        condition_prob = np.zeros([self.num_classes, self.num_classes])
        sample_per_cls = np.asarray([len(x) for x in self.gt_labels])
        rank_idx = np.argsort(-sample_per_cls)

        for i, cls_labels in enumerate(self.gt_labels):
            num = len(cls_labels)
            condition_prob[i] = np.sum(np.asarray(cls_labels), axis=0) / num

        sum_prob = np.sum(condition_prob, axis=0)
        need_sample = sample_per_cls / sum_prob
