#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2021-09-11 (year-month-day)

import torch
import torch.nn as nn

from sgld_nrg.resnet import ResNet

import torch.nn.functional as F


class EnergyModel(nn.Module):
    def logsumexp_logits(self, x):
        logits = self.forward(x)
        return torch.logsumexp(logits, dim=1)

    def predict_proba(self, x):
        return F.softmax(x, dim=1)


class ToyNet(EnergyModel):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (7, 7), (2, 2)),
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(800, 10)

    def forward(self, x):
        x_cnn = self.conv(x)
        x_flat = x_cnn.flatten(1)
        x_logit = self.fc(x_flat)
        return x_logit


class SimpleNet(EnergyModel):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (4, 4), (1, 1)),
            nn.BatchNorm2d(32),
            nn.Mish(),
            nn.Conv2d(32, 64, (4, 4), (1, 1)),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Mish(),
        )
        self.fc = nn.Linear(7744, 10)

    def forward(self, x):
        x_cnn = self.conv(x)
        x_flat = x_cnn.flatten(1)
        x_logit = self.fc(x_flat)
        return x_logit


class Resnet(EnergyModel):
    def __init__(self):
        super(Resnet, self).__init__()
        self.net = ResNet(in_channels=1, n_classes=10, depths=[2], activation="mish")

    def forward(self, x):
        return self.net(x)
