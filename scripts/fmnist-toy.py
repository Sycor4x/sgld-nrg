#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2021-09-07 (year-month-day)

"""
Implements Stochastic gradient Langevin dynamics for energy-based models, as per https://arxiv.org/pdf/1912.03263.pdf
"""


import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform as TorchUnif
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batch_size", type=int, help="how many samples in each training batch"
    )
    return parser.parse_args()


class SgldHarness(object):
    def __init__(
        self, net: nn.Module, replay_buffer, sgld_lr=1, sgld_step=20, noise=0.01
    ):
        assert isinstance(replay_buffer, ReplayBuffer)
        assert isinstance(sgld_lr, float) and sgld_lr > 0.0
        assert isinstance(sgld_step, int) and sgld_step > 0
        assert isinstance(noise, float) and noise > 0.0
        self.net = net
        self.sgld_lr = sgld_lr
        self.sgld_step = sgld_step
        self.noise = noise
        self.replay = replay_buffer
        self.sgld_optim = SGD(lr=sgld_lr, params=self.net.parameters())

    def step(self, x_hat, y):
        # TODO - do we need to know y to compute this? Need to review the paper... notation is confusing
        x_hat.requires_grad = True
        # TODO - get gradient for x wrt loss
        # https://discuss.pytorch.org/t/newbie-getting-the-gradient-with-respect-to-the-input/12709/7
        x_grad = torch.autograd.grad(self.net(x_hat), x_hat, create_graph=True)
        gradient_step = self.sgld_lr * x_grad
        noise = self.noise * torch.randn(self.replay.shape)
        x_hat = x_hat + gradient_step + noise
        x_hat[x_hat < self.replay.range[0]] = self.replay.range[0]
        x_hat[x_hat > self.replay.range[1]] = self.replay.range[1]
        return x_hat

    def __call__(self):
        x_hat = self.replay.sample()
        for _ in range(self.sgld_step):
            x_hat = self.step(x_hat, y)
        self.replay.append(x_hat)
        return x_hat


class ReplayBuffer(object):
    def __init__(
        self, data_shape, data_range, maxlen=10000, prob_reinitialize=0.05, seed=None
    ):
        # TODO - allow to set seed
        assert len(data_range) == 2
        assert data_range[0] != data_range[1]
        assert not np.isclose(data_range[0], data_range[1])
        assert all(isinstance(s, int) for s in data_shape)
        assert all(s > 0 for s in data_shape)
        assert isinstance(maxlen, int) and maxlen > 0
        assert 0.0 < prob_reinitialize < 1.0

        self.range = sorted(data_range)
        self.shape = data_shape
        self.buffer = []
        self.prob_reinitialize = prob_reinitialize
        self.maxlen = maxlen
        self.pointer = -1

    def append(self, new):
        if len(self.buffer) < self.maxlen:
            self.buffer.append(new)
        else:
            self.pointer += 1
            self.buffer[self.pointer % self.maxlen] = new

    def sample(self):
        if 0 < len(self.buffer) and np.random.rand() < 1.0 - self.prob_reinitialize:
            i = np.random.choice(range(len(self.buffer)))
            return self.buffer[i]
        else:
            return TorchUnif(self.range[0], self.range[1]).sample(*self.shape)


class AugmentedFashionMnistDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, item):
        x = self.x[item]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[item]


if __name__ == "__main__":
    user_args = parse_args()
    dest_dir = pathlib.Path("local_data")
    scale_xform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    fashion_train = FashionMNIST(
        dest_dir,
        train=True,
        transform=scale_xform,
        target_transform=None,
        download=True,
    )
    augmentation_xform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )
    fashion_train = AugmentedFashionMnistDataset(
        x=fashion_train.data, y=fashion_train.targets, transform=augmentation_xform
    )
    fashion_train = DataLoader(
        fashion_train, batch_size=user_args.batch_size, shuffle=True
    )

    fashion_test = FashionMNIST(
        dest_dir,
        train=False,
        transform=scale_xform,
        target_transform=None,
        download=True,
    )
    fashion_test = AugmentedFashionMnistDataset(
        x=fashion_test.data, y=fashion_test.targets, transform=None
    )
    fashion_test = DataLoader(fashion_test, batch_size=1000, shuffle=True)
