#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2021-09-11 (year-month-day)

"""
"""

import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli as TorchBernoulli
from torch.distributions.uniform import Uniform as TorchUnif


class SgldLogitEnergy(object):
    def __init__(
        self, net: nn.Module, replay_buffer, sgld_lr=1.0, sgld_step=20, noise=0.01
    ):
        assert isinstance(replay_buffer, ReplayBuffer)
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

    def get_energy(self, x_hat_):
        return -1.0 * self.net.logsumexp_logits(x_hat_)

    def get_grad(self, x_hat_):
        net_mode = self.net.training
        self.net.eval()
        # https://discuss.pytorch.org/t/newbie-getting-the-gradient-with-respect-to-the-input/12709/7
        x_hat_grad = torch.zeros_like(x_hat_)
        for j in range(x_hat_.size(0)):
            x_hat_j = x_hat_[j, ...].unsqueeze(0)
            nrg = self.get_energy(x_hat_j)
            grad_j = torch.autograd.grad(nrg, x_hat_j, create_graph=True)
            x_hat_grad[j, ...] = grad_j[0]
        self.net.training = net_mode
        return x_hat_grad

    def step(self, x_hat_):
        x_hat_grad = self.get_grad(x_hat_)
        gradient_step = self.sgld_lr * x_hat_grad
        noise = self.noise * torch.randn(x_hat_.shape)
        # this is gradient ASCENT because we want the samples to have high probability
        x_hat_ = x_hat_ + gradient_step + noise
        x_hat_[x_hat_ < min(self.replay.range)] = min(self.replay.range)
        x_hat_[x_hat_ > max(self.replay.range)] = max(self.replay.range)
        return x_hat_

    def __call__(self, batch_size):
        x_hat_ = self.replay.sample(batch_size)
        x_hat_.requires_grad = True
        for sgld_step_num in range(self.sgld_step):
            x_hat_ = self.step(x_hat_)
        x_hat_ = x_hat_.detach()
        self.replay.append(x_hat_)
        return x_hat_


class ReplayBuffer(object):
    def __init__(self, data_shape, data_range, maxlen=10000, prob_reinitialize=0.05):
        # TODO - allow to set seed
        assert len(data_range) == 2
        assert data_range[0] != data_range[1]
        assert all(isinstance(s, int) for s in data_shape)
        assert all(s > 0 for s in data_shape)
        assert isinstance(maxlen, int) and maxlen > 0
        assert 0.0 <= prob_reinitialize < 1.0

        self.range = data_range
        self.shape = data_shape
        self.buffer = torch.zeros((maxlen, *data_shape))
        self.prob_reinit = prob_reinitialize
        self.maxlen = maxlen
        self.pointer = 0
        self.length = 0

    def __len__(self):
        return self.length

    def append(self, new):
        into = torch.arange(0, new.size(0), dtype=torch.long) + self.pointer
        into %= self.maxlen
        self.buffer[into, ...] = new
        self.pointer += new.size(0)
        self.pointer %= self.maxlen
        self.length = min(self.maxlen, self.length + new.size(0))

    def sample(self, batch):
        assert isinstance(batch, int)
        assert batch < self.maxlen
        if 0 < self.length:
            ndx = torch.randint(len(self), size=(batch,))
            out = self.buffer[ndx, ...]
            mask = TorchBernoulli(probs=self.prob_reinit).sample((batch,))
            out[mask > 0.5, ...] = self.initialize(int(mask.sum()))
            return out
        else:
            return self.initialize(batch)

    def initialize(self, n):
        x_hat_ = TorchUnif(min(self.range), max(self.range)).sample((n, *self.shape))
        return x_hat_
