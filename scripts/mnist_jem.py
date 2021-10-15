#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2021-09-07 (year-month-day)

"""
Implements Stochastic gradient Langevin dynamics for energy-based models,
as per https://arxiv.org/pdf/1912.03263.pdf
"""

# TODO - periodically, rank all of the MCMC samples by their probability, and plot the highest-probability samples
# TODO -- implement checkpointing & automatic reversion to a saved model if some loss metric increases too much
# TODO -- implement 3-way split for MNIST (we have 2-way right now)

import datetime
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

from sgld_nrg.networks import Resnet, SimpleNet, ToyNet
from sgld_nrg.sgld import IndependentReplayBuffer, SgldLogitEnergy
from sgld_nrg.transform import AddGaussianNoise
from sgld_nrg.utils import estimate_time_remaining, get_accuracy, parse_args

if __name__ == "__main__":
    user_args = parse_args()
    data_dir = pathlib.Path("local_data")
    normalize_center = 0.5
    normalize_scale = 0.5
    scale_list = [
        transforms.ToTensor(),
        transforms.Normalize((normalize_center,), (normalize_scale,)),
    ]
    scale_xform = transforms.Compose(scale_list)
    augmentation_xform_list = []
    if user_args.translate > 0:
        augmentation_xform_list += [
            transforms.RandomCrop(
                (28, 28),
                padding=(user_args.translate, user_args.translate),
                fill=0,
                padding_mode="edge",
            )
        ]
    if user_args.input_noise > 0.0:
        augmentation_xform_list += [AddGaussianNoise(std=user_args.input_noise)]
    augmentation_xform = transforms.Compose(scale_list + augmentation_xform_list)
    print(f"We are using these transforms:\n{augmentation_xform}")
    if user_args.fashion:
        train = FashionMNIST(
            data_dir,
            train=True,
            transform=augmentation_xform,
            target_transform=None,
            download=True,
        )
        test = FashionMNIST(
            data_dir,
            train=False,
            transform=scale_xform,
            target_transform=None,
            download=True,
        )
    else:
        train = MNIST(
            data_dir,
            train=True,
            transform=augmentation_xform,
            target_transform=None,
            download=True,
        )
        test = MNIST(
            data_dir,
            train=False,
            transform=scale_xform,
            target_transform=None,
            download=True,
        )
    train = DataLoader(
        train,
        batch_size=user_args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=user_args.num_workers,
    )
    test = DataLoader(
        test,
        batch_size=1000,
        shuffle=True,
        drop_last=False,
        num_workers=user_args.num_workers,
    )

    if user_args.network == "simple":
        my_net = SimpleNet()
    elif user_args.network == "resnet":
        my_net = Resnet()
    elif user_args.network == "toy":
        my_net = ToyNet()
    else:
        raise ValueError(
            f"User argument to `--network` not recognized: {user_args.network}"
        )
    main_optim = Adam(my_net.parameters(), user_args.lr)

    torch_summary(my_net, (1, 28, 28))
    param_ct = sum(p.numel() for p in my_net.parameters() if p.requires_grad)
    print(f"The network has {param_ct:,} parameters.")
    my_buffer = IndependentReplayBuffer(
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
    if user_args.save.exists() and user_args.save.absolute().is_file():
        print(
            f"Loading saved model from {user_args.save}; continuing to train from this checkpoint..."
        )
        checkpoint = torch.load(user_args.save)
        my_net.load_state_dict(checkpoint["model_state_dict"])
        main_optim.load_state_dict(checkpoint["optimizer_state_dict"])
        my_sgld.replay.buffer = checkpoint["MCMC_samples"]
    else:
        print(
            f"No saved model {user_args.save} found; training the model from a random initialization..."
        )

    xe_loss_fn = nn.CrossEntropyLoss(reduction="mean")

    writer = SummaryWriter()

    sgld_train_buff = max(4, int(512 / user_args.batch_size + 0.5))
    sgld_train_total_buff = np.zeros(sgld_train_buff)
    sgld_train_xe_buff = np.zeros(sgld_train_buff)
    sgld_train_nrg_buff = np.zeros(sgld_train_buff)
    sgld_train_acc_buff = np.zeros(sgld_train_buff)
    sgld_train_pnrg_buff = np.zeros(sgld_train_buff)
    sgld_train_nnrg_buff = np.zeros(sgld_train_buff)
    sgld_sample_buff = torch.zeros(
        36, 1, 28, 28
    )  # fixed at 36 images, regardless of batch size
    jem_counter = 0
    total_loss = 0.0
    for epoch_num in range(user_args.n_epochs):
        my_sgld.sgld_step = min(50, my_sgld.sgld_step + 5)
        print(f"SGLD-train epoch {epoch_num} of {user_args.n_epochs}")
        start_time = datetime.datetime.now()
        for i, (x_train, y_train) in enumerate(train):
            my_net.train()
            jem_counter += x_train.size(0)
            y_logit = my_net(x_train)
            xe_loss = xe_loss_fn(y_logit, y_train)
            x_nrg = my_net.logsumexp_logits(x_train).mean()
            x_hat = my_sgld(user_args.batch_size)
            x_hat_nrg = my_net.logsumexp_logits(x_hat).mean()
            total_loss = xe_loss + x_nrg - x_hat_nrg
            total_loss.backward()
            main_optim.step()
            main_optim.zero_grad()
            train_acc = get_accuracy(y_logit, y_train)
            sgld_train_total_buff[i % sgld_train_buff] = total_loss.item()
            sgld_train_xe_buff[i % sgld_train_buff] = xe_loss.item()
            sgld_train_nrg_buff[i % sgld_train_buff] = (x_nrg - x_hat_nrg).item()
            sgld_train_pnrg_buff[i % sgld_train_buff] = x_nrg.item()
            sgld_train_nnrg_buff[i % sgld_train_buff] = x_hat_nrg.item()
            sgld_train_acc_buff[i % sgld_train_buff] = train_acc
            if x_hat.size(0) < sgld_sample_buff.size(0):
                into = torch.arange(0, x_hat.size(0)) + jem_counter
                into %= sgld_sample_buff.size(0)
                sgld_sample_buff[into, ...] = x_hat
            else:
                sgld_sample_buff = x_hat[: sgld_sample_buff.size(0), ...]
            if i % sgld_train_buff == 0:
                elapsed_s = (datetime.datetime.now() - start_time) / datetime.timedelta(
                    seconds=1
                )
                remaining_h = estimate_time_remaining(
                    elapsed_seconds=elapsed_s, batch_offset=i, num_batch=len(train)
                )
                print(
                    f"\tBatch {i} of {len(train)}; estimate {remaining_h:.2f} hours remaining in this epoch"
                )
                x_hat_grid = make_grid(
                    -1.0
                    * (
                        sgld_sample_buff * normalize_scale + normalize_center
                    ),  # reverse the scaling applied earlier for display purposes
                    nrow=int(np.sqrt(sgld_sample_buff.size(0)) + 0.5),
                )
                writer.add_image("mcmc_batch/x_hat", x_hat_grid, jem_counter)
                best_samples, worst_samples, mcmc_energy = my_sgld.summarize(
                    k=sgld_sample_buff.size(0), batch_size=100
                )
                writer.add_histogram(f"nrg/mcmc_energy", mcmc_energy, jem_counter)
                writer.add_image(
                    "top/best",
                    make_grid(
                        -1.0 * best_samples,
                        nrow=int(np.sqrt(sgld_sample_buff.size(0)) + 0.5),
                    ),
                    jem_counter,
                )
                writer.add_image(
                    "top/worst",
                    make_grid(
                        -1.0 * worst_samples,
                        nrow=int(np.sqrt(sgld_sample_buff.size(0)) + 0.5),
                    ),
                    jem_counter,
                )
                writer.add_scalar("loss/total", total_loss.item(), jem_counter)
                writer.add_scalar("loss/xe", sgld_train_xe_buff.mean(), jem_counter)
                writer.add_scalar(
                    "loss/\u0394nrg", (x_nrg - x_hat_nrg).item(), jem_counter
                )
                writer.add_scalar("loss/+nrg", x_nrg.item(), jem_counter)
                writer.add_scalar("loss/-nrg", x_hat_nrg.item(), jem_counter)
                writer.add_scalar("auxiliary/accuracy", train_acc, jem_counter)
                writer.add_scalar(
                    "auxiliary/seconds_per_instance",
                    elapsed_s / ((i + 1) * user_args.batch_size),
                    jem_counter,
                )
                for name, weight in my_net.named_parameters():
                    writer.add_histogram(f"param/{name}", weight, jem_counter)
                    writer.add_histogram(f"grad/{name}", weight.grad, jem_counter)

            if i > 0 and i % sgld_train_buff == 0:
                writer.add_scalar(
                    "loss/total", sgld_train_total_buff.mean(), jem_counter
                )
                writer.add_scalar("loss/xe", sgld_train_xe_buff.mean(), jem_counter)
                writer.add_scalar(
                    "loss/\u0394nrg", sgld_train_nrg_buff.mean(), jem_counter
                )
                writer.add_scalar("loss/+nrg", sgld_train_pnrg_buff.mean(), jem_counter)
                writer.add_scalar("loss/-nrg", sgld_train_nnrg_buff.mean(), jem_counter)
                writer.add_scalar(
                    "auxiliary/accuracy", sgld_train_acc_buff.mean(), jem_counter
                )
                writer.add_scalar("auxiliary/accuracy", train_acc, jem_counter)
        for j, (x_test, y_test) in enumerate(test):
            # TODO
            break
        save_dir = user_args.save.absolute().parent
        stem = user_args.save.absolute().stem
        fname = f"{stem}-epoch-{epoch_num}.pth"
        save_dest = save_dir.joinpath(fname)
        print(f"Saving model checkpoint to {save_dest}...")
        torch.save(
            {
                "epoch": epoch_num,
                "model_state_dict": my_net.state_dict(),
                "optimizer_state_dict": main_optim.state_dict(),
                "loss": total_loss,
                "MCMC_samples": my_sgld.replay.buffer,
            },
            str(save_dest),
        )
    writer.close()
