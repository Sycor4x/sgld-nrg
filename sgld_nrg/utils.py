#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2021-09-21 (year-month-day)

"""
"""
import argparse
import pathlib

import torch


def get_accuracy(y_hat, y):
    _, predicted = torch.max(y_hat.data, 1)
    correct = (predicted == y).sum().item()
    return correct / y.size(0)


def estimate_time_remaining(elapsed_seconds, batch_offset, num_batch):
    multiplier = (1.0 - (batch_offset + 1) / num_batch) / (batch_offset + 1) * num_batch
    return elapsed_seconds / 60.0 / 60.0 * multiplier


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
    fvalue = float(value)
    if fvalue <= 0.0:
        raise argparse.ArgumentTypeError(
            f"provided value {value} is not a positive float"
        )
    return fvalue


def nonnegative_float(value):
    fvalue = float(value)
    if fvalue < 0.0:
        raise argparse.ArgumentTypeError(f"provided value {value} is a negative float")
    return fvalue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--save",
        required=True,
        type=pathlib.Path,
        help="must provide a name for checkpoints; if the checkpoint already exists, the model is loaded",
    )
    parser.add_argument(
        "--pretrain_energy_penalty",
        default=0.0,
        type=nonnegative_float,
        help="how much to penalize the pre-training model using the energy; set to 0.0 (default) to disable",
    )
    parser.add_argument(
        "--num_workers",
        default=2,
        type=nonnegative_int,
        help="number of processes to spawn for DataLoader",
    )
    parser.add_argument(
        "--translate",
        default=0,
        type=nonnegative_int,
        help="randomly translates the image but up to this many pixels; set to 0 to disable",
    )
    parser.add_argument(
        "--input_noise",
        default=0.03,
        type=positive_float,
        help="the magnitude of 0-mean Gaussian noise applied to the MNIST images in each minibatch; set to 0 to disable",
    )
    parser.add_argument(
        "--network",
        default="simple",
        choices=["simple", "resnet", "toy"],
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
        default=15,
        type=positive_int,
        help="how many SGLD steps to take at each iteration",
    )
    parser.add_argument(
        "-b", "--batch_size_min", default=8 , type=positive_int, help="mini-batch size"
    )
    parser.add_argument(
         "--batch_size_max", default=256, type=positive_int, help="largest mini-batch size"
    )
    parser.add_argument(
        "-e",
        "--n_epochs",
        default=10,
        type=nonnegative_int,
        help="number of epochs of SGLD training",
    )
    return parser.parse_args()
