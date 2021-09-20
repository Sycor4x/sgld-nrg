#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2021-09-11 (year-month-day)

import torch
import torch.nn as nn

from sgld_nrg.resnet import small_resent, ResNet


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 4, 1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        self.fc = nn.Linear(7744, 10)

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
        self.net = ResNet(in_channels=1, n_classes=10, depths=[2], activation="elu")

    def forward(self, x):
        return self.net(x)

    def logsumexp_logits(self, x):
        logits = self.forward(x)
        return torch.logsumexp(logits, dim=1)
