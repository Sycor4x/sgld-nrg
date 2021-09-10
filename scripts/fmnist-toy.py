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
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--lr",
        default=1e-5,
        type=float,
        help="the learning rate for the optimizer",
    )
    parser.add_argument(
        "-e", "--n_epochs", default=1, type=int, help="number of epochs of training"
    )
    parser.add_argument(
        "-p",
        "--pre_epochs",
        default=15,
        type=int,
        help="how many epochs to train the network before using SGLD",
    )
    return parser.parse_args()


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1),
            # nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 4, 1),
            # nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2),
        )
        self.fc_proj = nn.Linear(7744, 128)
        self.fc_resid = nn.Sequential(
            nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            nn.ELU(),
            # nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ELU(),
        )
        self.fc_clf = nn.Linear(128, 10)

    def forward(self, x):
        x_cnn = self.conv(x)
        x_flat = x_cnn.flatten(1)
        x_proj = self.fc_proj(x_flat)
        x_resid = self.fc_resid(x_proj)
        x_logit = self.fc_clf(x_resid + x_proj)
        return x_logit

    def logsumexp_logits(self, x):
        logits = self.forward(x)
        return torch.logsumexp(logits, dim=1)


class SgldLogitHarness(object):
    def __init__(
        self, net: nn.Module, replay_buffer, sgld_lr=1.0, sgld_step=20, noise=0.01
    ):
        assert isinstance(replay_buffer, ReplayBuffer)
        assert isinstance(sgld_lr, float) and sgld_lr > 0.0
        assert isinstance(sgld_step, int) and sgld_step > 0
        assert isinstance(noise, float) and noise > 0.0
        assert noise <= sgld_lr
        self.net = net
        self.sgld_lr = sgld_lr
        self.sgld_step = sgld_step
        self.noise = noise
        self.replay = replay_buffer

    def __len__(self):
        return len(self.replay)

    def get_energy(self, x_hat_):
        return self.net.logsumexp_logits(x_hat_)

    def step(self, x_hat_):
        # TODO - get gradient for x wrt loss
        # https://discuss.pytorch.org/t/newbie-getting-the-gradient-with-respect-to-the-input/12709/7
        energy = self.get_energy(x_hat_)
        x_hat_grad = torch.autograd.grad(energy, x_hat_, create_graph=True)
        gradient_step = self.sgld_lr * x_hat_grad[0]
        noise = self.noise * torch.randn(x_hat_.shape)
        x_hat_ = x_hat_ + gradient_step + noise
        x_hat_[x_hat_ < min(self.replay.range)] = min(self.replay.range)
        x_hat_[x_hat_ > max(self.replay.range)] = max(self.replay.range)
        return x_hat_

    def __call__(self):
        x_hat_ = self.replay.sample()
        x_hat_.requires_grad = True
        for sgld_step_num in range(self.sgld_step):
            x_hat_ = self.step(x_hat_)
        x_hat_ = x_hat_.detach()
        self.replay.append(x_hat_)
        return x_hat_


class SgldClassHarness(SgldLogitHarness):
    def get_energy(self, x_hat_):
        y_hat_ = np.random.choice(range(10))
        return self.net(x_hat_)[y_hat_]


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
        assert 0.0 <= prob_reinitialize < 1.0

        self.range = data_range
        self.shape = data_shape
        self.buffer = []
        self.prob_reinitialize = prob_reinitialize
        self.maxlen = maxlen
        self.pointer = -1

    def __len__(self):
        return len(self.buffer)

    def append(self, new):
        if len(self.buffer) < self.maxlen:
            self.buffer.append(new)
        else:
            self.pointer += 1
            self.buffer[self.pointer % self.maxlen] = new

    def sample(self):
        if 0 < len(self.buffer) and np.random.rand() < 1.0 - self.prob_reinitialize:
            ndx = np.random.choice(range(len(self)))
            return self.buffer[ndx]
        else:
            return TorchUnif(min(self.range), max(self.range)).sample(self.shape)


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, t):
        return t + torch.randn(t.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"


if __name__ == "__main__":
    user_args = parse_args()
    dest_dir = pathlib.Path("local_data")
    scale_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    scale_xform = transforms.Compose(scale_list)
    augmentation_xform = transforms.Compose(
        # [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
        # + scale_list
        scale_list
        + [AddGaussianNoise(std=0.03)]
    )
    fashion_train_d = MNIST(
        dest_dir,
        train=True,
        transform=augmentation_xform,
        target_transform=None,
        download=True,
    )
    # TODO - create a transform for adding 0-mean noise with a chosen stddev
    fashion_train = DataLoader(fashion_train_d, batch_size=1, shuffle=True)
    fashion_pre_train = DataLoader(fashion_train_d, batch_size=100, shuffle=True)

    fashion_test = MNIST(
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

    my_buffer = ReplayBuffer(
        data_shape=(1, 1, 28, 28),
        data_range=(-1, 1),
        maxlen=10000,
        prob_reinitialize=0.05,
    )
    my_sgld = SgldClassHarness(
        net=my_net, replay_buffer=my_buffer, sgld_lr=1.0, noise=1e-2, sgld_step=50
    )
    xe_loss_fn = nn.CrossEntropyLoss()
    main_optim = Adam(my_net.parameters(), user_args.lr, weight_decay=1e-5)
    pre_train_optim = Adam(my_net.parameters(), 1e-4)
    writer = SummaryWriter()

    pre_train_buff_size = 60
    pre_train_buff = np.zeros(pre_train_buff_size)
    pre_val_buff = np.zeros(len(fashion_test))
    pre_train_counter = -1
    pointer = -1
    for pre_train_epoch in range(user_args.pre_epochs):
        print(f"Pre-train epoch {pre_train_epoch} of {user_args.pre_epochs}")
        for j, (x_val, y_val) in enumerate(fashion_test):
            my_net.eval()
            y_logit = my_net(x_val)
            xe_loss = xe_loss_fn(y_logit, y_val)
            pre_val_buff[j] = xe_loss.item()
            writer.add_scalar(
                "Pretrain/xe_loss_test", pre_val_buff.mean(), pre_train_counter
            )
        for i, (x_train, y_train) in enumerate(fashion_pre_train):
            my_net.train()
            pre_train_counter += x_train.size(0)
            pre_train_optim.zero_grad()
            y_logit = my_net(x_train)
            xe_loss = xe_loss_fn(y_logit, y_train)
            xe_loss.backward()
            pre_train_optim.step()
            pointer += 1
            pre_train_buff[pointer % pre_train_buff_size] = xe_loss.item()
            if i % pre_train_buff_size == 0 and i > 0:
                writer.add_scalar(
                    "Pretrain/xe_loss_train", pre_train_buff.mean(), pre_train_counter
                )

    print("Initializing the SGLD replay buffer")
    for _ in range(1000):
        my_sgld()
    my_net.conv.requires_grad_(False)

    sgld_train_buff_size = 100
    sgld_train_total_buff = np.zeros(sgld_train_buff_size)
    sgld_train_xe_buff = np.zeros(sgld_train_buff_size)
    sgld_train_nrg_buff = np.zeros(sgld_train_buff_size)
    x_hat_buff = torch.zeros((100, 1, 28, 28))
    jem_counter = -1
    for epoch_num in range(user_args.n_epochs):
        print(f"SGLD-train epoch {epoch_num} of {user_args.n_epochs}")
        for i, (x_train, y_train) in enumerate(fashion_train):
            my_net.train()
            pre_train_optim.zero_grad()
            jem_counter += x_train.size(0)
            y_logit = my_net(x_train)
            xe_loss = xe_loss_fn(y_logit, y_train)
            x_nrg = my_net.logsumexp_logits(x_train)
            x_hat = my_sgld()
            x_hat_nrg = my_net.logsumexp_logits(x_hat)
            total_loss = xe_loss + x_nrg - x_hat_nrg
            total_loss.backward()
            pre_train_optim.step()
            # main_optim.step()
            # main_optim.zero_grad()
            sgld_train_total_buff[i % sgld_train_buff_size] = total_loss.item()
            sgld_train_xe_buff[i % sgld_train_buff_size] = xe_loss.item()
            sgld_train_nrg_buff[i % sgld_train_buff_size] = (x_nrg - x_hat_nrg).item()

            x_hat_buff[i % sgld_train_buff_size, ...] = x_hat
            grid = make_grid(x_hat_buff)

            writer.add_scalar("JEM/total", total_loss.item(), jem_counter)
            writer.add_scalar("JEM/xe", xe_loss.item(), jem_counter)
            writer.add_scalar("JEM/nrg", (x_nrg - x_hat_nrg).item(), jem_counter)

            if i % sgld_train_buff_size == 0 and i > 0:
                writer.add_scalar(
                    "JEM/total", sgld_train_total_buff.mean(), jem_counter
                )
                writer.add_scalar("JEM/xe", sgld_train_xe_buff.mean(), jem_counter)
                writer.add_scalar("JEM/nrg", sgld_train_nrg_buff.mean(), jem_counter)
                writer.add_image("JEM/x_hat", grid, jem_counter)
