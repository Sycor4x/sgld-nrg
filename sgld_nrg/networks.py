#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2021-09-11 (year-month-day)

import torch
import torch.nn as nn

from sgld_nrg.resnet import small_resent


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


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.net = small_resent(in_channels=1, n_classes=10, activation="elu")

    def forward(self, x):
        return self.net(x)

    def logsumexp_logits(self, x):
        logits = self.forward(x)
        return torch.logsumexp(logits, dim=1)
