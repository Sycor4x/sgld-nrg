#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2021-09-11 (year-month-day)

import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 4, 1),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(7744),
            nn.ELU(),
            nn.Linear(7744, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(),
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


class ResidNet(nn.Module):
    # Some helpful info about padding mechanics: https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121
    def __init__(self):
        super(ResidNet, self).__init__()
        # TODO - residual convolution layers
        self.conv_inp = nn.Conv2d(1, 32, (4, 4), (1, 1))
        self.conv_resid_1 = nn.Sequential(
            nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        self.conv_resid_2 = nn.Sequential(
            nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, (3, 3), (1, 1), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        self.down = nn.MaxPool2d(2)
        # self.down = nn.Conv2d(32, 8, (1, 1), (1, 1))
        self.fc_proj = nn.Linear(4608, 128, bias=False)
        self.fc_resid_1 = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 128, bias=False),
        )
        self.fc_resid_2 = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 128, bias=False),
        )
        self.fc_clf = nn.Linear(128, 10)

    def forward(self, x):
        x_cnn_in = self.conv_inp(x)
        x_cnn_1 = self.conv_resid_1(x_cnn_in) + x_cnn_in
        x_cnn_2 = self.conv_resid_2(x_cnn_1) + x_cnn_1
        x_pool = self.down(x_cnn_2)
        x_flat = x_pool.flatten(1)
        x_proj = self.fc_proj(x_flat)
        x_fc_1 = self.fc_resid_1(x_proj) + x_proj
        x_fc_2 = self.fc_resid_2(x_fc_1) + x_fc_1
        x_logit = self.fc_clf(x_fc_2)
        return x_logit

    def logsumexp_logits(self, x):
        logits = self.forward(x)
        return torch.logsumexp(logits, dim=1)
