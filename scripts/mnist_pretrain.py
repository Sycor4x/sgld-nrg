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
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

from sgld_src.transform import AddGaussianNoise
from sgld_src.networks import SimpleNet
from sgld_src.sgld import ClassReplayBuffer, SgldLogitHarness


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
    print(f"The network has {param_ct:,} parameters.")

    my_buffer = ClassReplayBuffer(
        data_shape=(1, 1, 28, 28),
        data_range=(-1.0, 1.0),
        maxlen=10000,
        prob_reinitialize=0.1,
    )
    my_sgld = SgldLogitHarness(
        net=my_net, replay_buffer=my_buffer, sgld_lr=1e0, noise=1e-2, sgld_step=50
    )
    xe_loss_fn = nn.CrossEntropyLoss()
    pre_train_optim = Adam(my_net.parameters(), 3e-4)
    main_optim = Adam(my_net.parameters(), user_args.lr, weight_decay=1e-5)

    pre_train_buff_size = 60
    pre_train_buff = np.zeros(pre_train_buff_size)
    pre_val_buff = np.zeros(len(fashion_test))
    pre_train_counter = -1
    pointer = -1
    writer = SummaryWriter()
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
    grid_buff_size = 64
    sgld_train_buff_size = grid_buff_size
    x_hat_buff = torch.zeros((grid_buff_size, 1, 28, 28))
    for k in range(grid_buff_size):
        x_hat_buff[k, ...], _ = my_sgld()
    grid = make_grid(x_hat_buff)
    writer.add_image("JEM/x_hat", grid, 0)
    # my_net.conv.requires_grad_(False)

    sgld_train_total_buff = np.zeros(sgld_train_buff_size)
    sgld_train_xe_buff = np.zeros(sgld_train_buff_size)
    sgld_train_nrg_buff = np.zeros(sgld_train_buff_size)
    jem_counter = -1
    for epoch_num in range(user_args.n_epochs):
        print(f"SGLD-train epoch {epoch_num} of {user_args.n_epochs}")
        for i, (x_train, y_train) in enumerate(fashion_train):
            my_net.train()
            # pre_train_optim.zero_grad()
            main_optim.zero_grad()
            jem_counter += x_train.size(0)
            y_logit = my_net(x_train)
            xe_loss = xe_loss_fn(y_logit, y_train)
            x_nrg = my_net.logsumexp_logits(x_train)
            # x_nrg = my_net(x_train)[:, y_train]
            x_hat, y_hat = my_sgld()
            # x_hat_nrg = my_net(x_hat)[:, y_hat]
            x_hat_nrg = my_net.logsumexp_logits(x_hat)
            total_loss = xe_loss + x_nrg - x_hat_nrg
            total_loss.backward()
            # pre_train_optim.step()
            main_optim.step()
            sgld_train_total_buff[i % sgld_train_buff_size] = total_loss.item()
            sgld_train_xe_buff[i % sgld_train_buff_size] = xe_loss.item()
            sgld_train_nrg_buff[i % sgld_train_buff_size] = (x_nrg - x_hat_nrg).item()

            x_hat_buff[i % sgld_train_buff_size, ...] = x_hat

            writer.add_scalar("JEM/total", total_loss.item(), jem_counter)
            writer.add_scalar("JEM/xe", xe_loss.item(), jem_counter)
            writer.add_scalar("JEM/\u0394nrg", (x_nrg - x_hat_nrg).item(), jem_counter)

            if i % sgld_train_buff_size == 0 and i > 0:
                writer.add_scalar(
                    "JEM/total", sgld_train_total_buff.mean(), jem_counter
                )
                writer.add_scalar("JEM/xe", sgld_train_xe_buff.mean(), jem_counter)
                writer.add_scalar("JEM/nrg", sgld_train_nrg_buff.mean(), jem_counter)
                x_hat_grid = make_grid(x_hat_buff)
                writer.add_image("JEM/x_hat", x_hat_grid, jem_counter)
