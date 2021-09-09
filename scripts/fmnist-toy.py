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
        "-b",
        "--batch_size",
        default=64,
        type=int,
        required=True,
        help="how many samples in each training batch",
    )
    return parser.parse_args()


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            # nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 1),
            # nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(9216, 128),
            # nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x_cnn = self.conv(x)
        x_flat = x_cnn.flatten(1)
        x_logit = self.fc(x_flat)
        return x_logit

    def logsumexp_logits(self, x):
        logits = self.forward(x)
        return torch.logsumexp(logits, dim=1)


class SgldHarness(object):
    def __init__(
        self, net: nn.Module, replay_buffer, sgld_lr=1., sgld_step=20, noise=0.01
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

    def step(self, x_hat):
        # TODO - get gradient for x wrt loss
        # https://discuss.pytorch.org/t/newbie-getting-the-gradient-with-respect-to-the-input/12709/7
        energy = self.net.logsumexp_logits(x_hat)
        x_grad = torch.autograd.grad(energy, x_hat, create_graph=True)[0]
        gradient_step = self.sgld_lr * x_grad
        noise = self.noise * torch.randn(self.replay.shape)
        x_hat = x_hat + gradient_step + noise
        x_hat[x_hat < min(self.replay.range)] = min(self.replay.range)
        x_hat[x_hat > max(self.replay.range)] = max(self.replay.range)
        return x_hat

    def __call__(self):
        x_hat = self.replay.sample()
        x_hat.requires_grad = True
        for sgld_step_num in range(self.sgld_step):
            x_hat = self.step(x_hat)
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

        self.range = data_range
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
            ndx = np.random.choice(range(len(self.buffer)))
            return self.buffer[ndx]
        else:
            return TorchUnif(min(self.range), max(self.range)).sample(self.shape)


if __name__ == "__main__":
    user_args = parse_args()
    dest_dir = pathlib.Path("local_data")
    scale_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    scale_xform = transforms.Compose(scale_list)
    augmentation_xform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
        + scale_list
    )
    fashion_train = FashionMNIST(
        dest_dir,
        train=True,
        transform=augmentation_xform,
        target_transform=None,
        download=True,
    )
    # TODO - create a transform for adding 0-mean noise with a chosen stddev
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
    fashion_test = DataLoader(fashion_test, batch_size=1000, shuffle=True)

    my_net = SimpleNet()
    param_ct = sum(p.numel() for p in my_net.parameters() if p.requires_grad)
    # print(my_net)
    print(f"The network has {param_ct:,} parameters.")

    my_buffer = ReplayBuffer(data_shape=(1,1,28,28), data_range=(-1,1))
    my_sgld = SgldHarness(net=my_net,replay_buffer=my_buffer)

    for i, (x, y) in enumerate(fashion_train):
        y_logit = my_net(x)
        print(y_logit)
        nrg = my_net.logsumexp_logits(x)
        print(nrg)
        my_sgld()
        break
        # TODO asdf
