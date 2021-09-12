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
            # nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 4, 1),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            # nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Linear(7744, 128),
            # nn.BatchNorm1d(128),
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
    def __init__(self):
        super(ResidNet, self).__init__()
        # TODO - residual convolution layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1),
            # nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 4, 1),
            nn.MaxPool2d(2),
            # nn.BatchNorm2d(64),
        )
        self.fc_proj = nn.Linear(7744, 128)
        self.fc_resid = nn.Sequential(
            # nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            # nn.BatchNorm1d(128),
            nn.Linear(128, 128),
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
