#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2021-09-11 (year-month-day)

"""
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform as TorchUnif


class SgldLogitHarness(object):
    def __init__(
        self, net: nn.Module, replay_buffer, sgld_lr=1.0, sgld_step=20, noise=0.01
    ):
        assert isinstance(replay_buffer, ClassReplayBuffer)
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

    def get_energy(self, x_hat_, y_hat_):
        return -1.0 * self.net.logsumexp_logits(x_hat_)

    def step(self, x_hat_, y_hat_):
        # TODO - get gradient for x wrt loss
        # https://discuss.pytorch.org/t/newbie-getting-the-gradient-with-respect-to-the-input/12709/7
        energy = self.get_energy(x_hat_, y_hat_)
        x_hat_grad = torch.autograd.grad(energy, x_hat_, create_graph=True)
        gradient_step = self.sgld_lr * x_hat_grad[0]
        noise = self.noise * torch.randn(x_hat_.shape)
        # this is gradient ASCENT because we want the samples to have high probability
        x_hat_ = x_hat_ + gradient_step + noise
        x_hat_[x_hat_ < min(self.replay.range)] = min(self.replay.range)
        x_hat_[x_hat_ > max(self.replay.range)] = max(self.replay.range)
        return x_hat_

    def __call__(self):
        x_hat_, y_hat_ = self.replay.sample()
        x_hat_.requires_grad = True
        for sgld_step_num in range(self.sgld_step):
            x_hat_ = self.step(x_hat_, y_hat_)
        x_hat_ = x_hat_.detach()
        self.replay.append((x_hat_, y_hat_))
        return x_hat_, y_hat_


class SgldClassHarness(SgldLogitHarness):
    def get_energy(self, x_hat_, y_hat_):
        logits = self.net(x_hat_)
        return -1.0 * logits[:, y_hat_]


class ClassReplayBuffer(object):
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
            x_hat_ = TorchUnif(min(self.range), max(self.range)).sample(self.shape)
            y_hat_ = np.random.choice(range(10))
            return x_hat_, y_hat_
