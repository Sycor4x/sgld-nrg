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
from torchsummary import summary as torch_summary
from torchvision import transforms
from torchvision.datasets import FashionMNIST, MNIST
from torchvision.utils import make_grid

from sgld_nrg.networks import Resnet, SimpleNet
from sgld_nrg.sgld import ReplayBuffer, SgldLogitEnergy
from sgld_nrg.transform import AddGaussianNoise


def nonnegative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(
            f"provided value {value} is not a positive int"
        )
    return ivalue


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"provided value {value} is not a positive int"
        )
    return ivalue


def positive_float(value):
    ivalue = float(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"provided value {value} is not a positive float"
        )
    return ivalue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_noise",
        default=0.03,
        type=positive_float,
        help="the magnitude of 0-mean Gaussian noise applied to the MNIST images in each minibatch",
    )
    parser.add_argument(
        "--network",
        default="residual",
        choices=["simple", "residual"],
        help="which neural network architecture to use for training; see networks.py",
    )
    parser.add_argument(
        "--sgld_lr",
        default=1.0,
        type=positive_float,
        help="the size of the SGLD gradient step",
    )
    parser.add_argument(
        "--sgld_noise",
        default=1e-2,
        type=positive_float,
        help="the standard deviation of the gaussian noise in SGLD",
    )
    parser.add_argument(
        "-F",
        "--fashion",
        action="store_true",
        help="set this flag to train with FashionMNIST instead of the MNIST digits",
    )
    parser.add_argument(
        "-r",
        "--lr",
        default=1e-5,
        type=float,
        help="the learning rate for the optimizer",
    )
    parser.add_argument(
        "--replay_buff",
        default=10000,
        type=positive_int,
        help="How many chains to store in the SGLD MCMC buffer",
    )
    parser.add_argument(
        "--prob_reinit",
        default=0.05,
        type=positive_float,
        help="The probability of reinitializing an SGLD MCMC chain",
    )
    parser.add_argument(
        "--sgld_steps",
        default=20,
        type=positive_int,
        help="how many SGLD steps to take at each iteration",
    )
    parser.add_argument(
        "-b", "--batch_size", default=64, type=positive_int, help="mini-batch size"
    )
    parser.add_argument(
        "-e",
        "--n_epochs",
        default=10,
        type=nonnegative_int,
        help="number of epochs of SGLD training",
    )
    parser.add_argument(
        "-p",
        "--pre_epochs",
        default=0,
        type=nonnegative_int,
        help="how many epochs to train the network as an ordinary classifier before using SGLD",
    )
    return parser.parse_args()


def get_accuracy(y_hat, y):
    _, predicted = torch.max(y_hat.data, 1)
    correct = (predicted == y).sum().item()
    return correct / y.size(0)


if __name__ == "__main__":
    user_args = parse_args()
    dest_dir = pathlib.Path("local_data")
    normalize_center = 0.5
    normalize_scale = 0.5
    scale_list = [
        transforms.ToTensor(),
        transforms.Normalize((normalize_center,), (normalize_scale,)),
    ]
    scale_xform = transforms.Compose(scale_list)
    augmentation_xform_list = scale_list.copy()
    augmentation_xform_list += [
        transforms.RandomCrop((28, 28), padding=(2, 2), fill=0, padding_mode="edge")
    ]
    if user_args.input_noise > 0.0:
        augmentation_xform_list += [AddGaussianNoise(std=user_args.input_noise)]
    augmentation_xform = transforms.Compose(augmentation_xform_list)
    if user_args.fashion:
        train = FashionMNIST(
            dest_dir,
            train=True,
            transform=augmentation_xform,
            target_transform=None,
            download=True,
        )
        test = FashionMNIST(
            dest_dir,
            train=False,
            transform=scale_xform,
            target_transform=None,
            download=True,
        )
    else:
        train = MNIST(
            dest_dir,
            train=True,
            transform=augmentation_xform,
            target_transform=None,
            download=True,
        )
        test = MNIST(
            dest_dir,
            train=False,
            transform=scale_xform,
            target_transform=None,
            download=True,
        )
    train = DataLoader(
        train, batch_size=user_args.batch_size, shuffle=True, drop_last=True
    )
    test = DataLoader(test, batch_size=1000, shuffle=True)

    if user_args.network == "simple":
        my_net = SimpleNet()
    elif user_args.network == "residual":
        my_net = Resnet()
    else:
        raise ValueError(
            f"User argument to `--network` not recognized: {user_args.network}"
        )

    torch_summary(my_net, (1, 28, 28))
    param_ct = sum(p.numel() for p in my_net.parameters() if p.requires_grad)
    print(f"The network has {param_ct:,} parameters.")

    my_buffer = ReplayBuffer(
        data_shape=(1, 28, 28),
        data_range=(-1.0, 1.0),
        maxlen=user_args.replay_buff,
        prob_reinitialize=user_args.prob_reinit,
    )
    my_sgld = SgldLogitEnergy(
        net=my_net,
        replay_buffer=my_buffer,
        sgld_lr=user_args.sgld_lr,
        noise=user_args.sgld_noise,
        sgld_step=user_args.sgld_steps,
    )
    xe_loss_fn = nn.CrossEntropyLoss(reduction="mean")
    pre_train_optim = Adam(my_net.parameters(), 3.0 * user_args.lr)
    main_optim = Adam(my_net.parameters(), user_args.lr)

    pre_train_buff_size = max(1, int(512 / user_args.batch_size + 0.5))
    pre_train_buff_xe = np.zeros(pre_train_buff_size)
    pre_train_buff_acc = np.zeros(pre_train_buff_size)
    pre_train_buff_pnrg = np.zeros(pre_train_buff_size)
    pre_val_buff = np.zeros(len(test))
    pre_val_buff_acc = np.zeros(len(test))
    pre_val_buff_pnrg = np.zeros(len(test))
    pre_train_counter = 0
    pointer = 0
    writer = SummaryWriter()
    for pre_train_epoch in range(user_args.pre_epochs):
        print(f"Pre-train epoch {pre_train_epoch} of {user_args.pre_epochs}")
        for i, (x_train, y_train) in enumerate(train):
            my_net.train()
            pre_train_counter += x_train.size(0)
            pre_train_optim.zero_grad()
            y_logit = my_net(x_train)
            xe_loss = xe_loss_fn(y_logit, y_train)
            xe_loss.backward()
            pnrg = my_net.logsumexp_logits(x_train).mean()
            pre_train_optim.step()
            pre_train_buff_acc[pointer % pre_train_buff_size] = get_accuracy(
                y_logit, y_train
            )
            pre_train_buff_xe[pointer % pre_train_buff_size] = xe_loss.item()
            pre_train_buff_pnrg[pointer % pre_train_buff_size] = pnrg
            pointer += 1
            if i % pre_train_buff_size == 0 and i > 0:
                writer.add_scalar(
                    "Pre/accuracy/train", pre_train_buff_acc.mean(), pre_train_counter
                )
                writer.add_scalar(
                    "Pre/xe_loss/train", pre_train_buff_xe.mean(), pre_train_counter
                )
                writer.add_scalar("+nrg", pre_train_buff_pnrg.mean(), pre_train_counter)
                for name, weight in my_net.named_parameters():
                    writer.add_histogram(f"param/{name}", weight, pre_train_counter)
                    writer.add_histogram(f"grad/{name}", weight.grad, pre_train_counter)

        for j, (x_val, y_val) in enumerate(test):
            my_net.eval()
            y_logit = my_net(x_val)
            xe_loss = xe_loss_fn(y_logit, y_val)
            pre_val_buff[j] = xe_loss.item()
            pre_val_buff_acc[j] = get_accuracy(y_logit, y_val)
            pre_val_buff_pnrg[j] = my_net.logsumexp_logits(x_val).mean()
            writer.add_scalar(
                "Pre/accuracy/val", pre_val_buff_acc.mean(), pre_train_counter
            )
            writer.add_scalar("Pre/xe_loss/val", pre_val_buff.mean(), pre_train_counter)
            writer.add_scalar(
                "Pre/+nrg/val", pre_val_buff_pnrg.mean(), pre_train_counter
            )

    sgld_train_buff = max(1, int(512 / user_args.batch_size + 0.5))
    sgld_train_total_buff = np.zeros(sgld_train_buff)
    sgld_train_xe_buff = np.zeros(sgld_train_buff)
    sgld_train_nrg_buff = np.zeros(sgld_train_buff)
    sgld_train_acc_buff = np.zeros(sgld_train_buff)
    sgld_train_pnrg_buff = np.zeros(sgld_train_buff)
    sgld_train_nnrg_buff = np.zeros(sgld_train_buff)
    jem_counter = 0
    for epoch_num in range(user_args.n_epochs):
        print(f"SGLD-train epoch {epoch_num} of {user_args.n_epochs}")
        for i, (x_train, y_train) in enumerate(train):
            my_net.train()
            main_optim.zero_grad()
            jem_counter += x_train.size(0)
            y_logit = my_net(x_train)
            xe_loss = xe_loss_fn(y_logit, y_train)
            x_nrg = my_net.logsumexp_logits(x_train).mean()
            x_hat = my_sgld(user_args.batch_size)
            x_hat_nrg = my_net.logsumexp_logits(x_hat).mean()
            total_loss = xe_loss + x_nrg - x_hat_nrg
            total_loss.backward()
            main_optim.step()
            train_acc = get_accuracy(y_logit, y_train)
            sgld_train_total_buff[i % sgld_train_buff] = total_loss.item()
            sgld_train_xe_buff[i % sgld_train_buff] = xe_loss.item()
            sgld_train_nrg_buff[i % sgld_train_buff] = (x_nrg - x_hat_nrg).item()
            sgld_train_pnrg_buff[i % sgld_train_buff] = x_nrg.item()
            sgld_train_nnrg_buff[i % sgld_train_buff] = x_hat_nrg.item()
            sgld_train_acc_buff[i % sgld_train_buff] = train_acc

            if i % sgld_train_buff == 0:
                print(f"\tBatch {i} of {len(train)}")
                x_hat_grid = make_grid(
                    normalize_scale * x_hat + normalize_center,
                    nrow=min(8, int(np.sqrt(user_args.batch_size))),
                )  # reverse the scaling appplied earlier
                writer.add_image("JEM/x_hat", x_hat_grid, jem_counter)
                writer.add_scalar("JEM/total", total_loss.item(), jem_counter)
                writer.add_scalar("JEM/xe", sgld_train_xe_buff.mean(), jem_counter)
                writer.add_scalar("JEM/\u0394nrg", xe_loss.item(), jem_counter)
                writer.add_scalar("JEM/+nrg", x_nrg.item(), jem_counter)
                writer.add_scalar("+nrg", x_nrg.item(), jem_counter)
                writer.add_scalar("JEM/-nrg", x_hat_nrg.item(), jem_counter)
                writer.add_scalar("JEM/accuracy", train_acc, jem_counter)
                for name, weight in my_net.named_parameters():
                    writer.add_histogram(f"param/{name}", weight, pre_train_counter)
                    writer.add_histogram(f"grad/{name}", weight.grad, pre_train_counter)

            if i > 0 and i % sgld_train_buff == 0:
                writer.add_scalar(
                    "JEM/total", sgld_train_total_buff.mean(), jem_counter
                )
                writer.add_scalar("JEM/xe", sgld_train_xe_buff.mean(), jem_counter)
                writer.add_scalar(
                    "JEM/\u0394nrg", sgld_train_nrg_buff.mean(), jem_counter
                )
                writer.add_scalar("JEM/+nrg", sgld_train_pnrg_buff.mean(), jem_counter)
                writer.add_scalar("+nrg", sgld_train_pnrg_buff.mean(), jem_counter)
                writer.add_scalar("JEM/-nrg", sgld_train_nnrg_buff.mean(), jem_counter)
                writer.add_scalar(
                    "JEM/accuracy", sgld_train_acc_buff.mean(), jem_counter
                )
