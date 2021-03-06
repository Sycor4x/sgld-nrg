#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2021-09-11 (year-month-day)

"""
Implements SGLD to sample from the input data distribution, and various buffers to
store the MCMC samples.
"""

import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli as TorchBernoulli
from torch.distributions.uniform import Uniform as TorchUnif
import numpy as np
from sgld_nrg.networks import EnergyModel


def make_batch(iterable, n=1):
    assert n > 0
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


class SgldLogitEnergy(object):
    def __init__(
        self, net: EnergyModel, replay_buffer, sgld_lr=1.0, sgld_step=20, noise=0.01
    ):
        assert isinstance(replay_buffer, RingReplayBuffer)
        assert isinstance(sgld_lr, float) and sgld_lr > 0.0
        assert isinstance(sgld_step, int) and sgld_step > 0
        assert isinstance(noise, float) and noise > 0.0
        assert noise <= sgld_lr
        self.net = net
        self.sgld_lr = sgld_lr
        self.sgld_step = sgld_step
        self.noise = noise
        self.replay = replay_buffer

    def get_energy(self, x_hat_):
        return -1.0 * self.net.logsumexp_logits(x_hat_)

    def get_grad(self, x_hat_):
        # https://discuss.pytorch.org/t/newbie-getting-the-gradient-with-respect-to-the-input/12709/7
        net_mode = self.net.training
        self.net.eval()
        x_hat_split = torch.tensor_split(x_hat_, x_hat_.size(0))
        nrg_split = (self.get_energy(x) for x in x_hat_split)
        grad_j = torch.autograd.grad(nrg_split, x_hat_split, create_graph=False)
        x_hat_grad = torch.stack(grad_j, 0).squeeze(1)
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

    @torch.no_grad()
    def summarize(self, k=36, batch_size=100):
        """
        Finds the best samples in the MCMC buffer and returns them.
        :param k: positive int - selects the k samples with the largest logsumexp(logit)
        :param batch_size: positive int - how many samples to push through the network at a time
        :return: torch.Tensor
        """
        assert k > 0 and isinstance(k, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        scores = torch.zeros(self.replay.maxlen)
        net_mode = self.net.training
        self.net.eval()
        for ndx in make_batch(range(self.replay.maxlen), batch_size):
            batch = self.replay.buffer[ndx, ...]
            scores[ndx] = self.net.logsumexp_logits(batch)
        ndx_sort = torch.argsort(scores)
        best = self.replay.buffer[ndx_sort[:k], ...]
        worst = self.replay.buffer[ndx_sort[-k:], ...]
        self.net.training = net_mode
        return best, worst, scores


class RingReplayBuffer(object):
    def __init__(self, data_shape, data_range, maxlen=10000, prob_reinitialize=0.05):
        """
        The ReplayBuffer is a ring buffer.
        :param data_shape: the shape of a sample
        :param data_range: the upper and lower values of the uniform random noise
        :param maxlen: number of samples to maintain
        :param prob_reinitialize: probability that a chain is initialized with random
            noise
        """
        # TODO -- set this up to store & retrieve a label alongside the generated images
        assert len(data_range) == 2
        assert data_range[0] != data_range[1]
        assert all(isinstance(s, int) for s in data_shape)
        assert all(s > 0 for s in data_shape)
        assert isinstance(maxlen, int) and maxlen > 0
        assert 0.0 <= prob_reinitialize < 1.0

        self.range = data_range
        self.shape = data_shape
        self.prob_reinit = prob_reinitialize
        self.maxlen = maxlen
        self.pointer = 0
        self.buffer = self.initialize(maxlen)

    def append(self, new):
        into = torch.arange(0, new.size(0), dtype=torch.long) + self.pointer
        into %= self.maxlen
        self.buffer[into, ...] = new
        self.pointer += new.size(0)
        self.pointer %= self.maxlen

    def sample(self, batch_size):
        assert isinstance(batch_size, int)
        assert batch_size < self.maxlen
        ndx = torch.randint(self.maxlen, size=(batch_size,))
        x = self.buffer[ndx, ...]
        x = self.maybe_reinitialize(x=x)
        return x

    def maybe_reinitialize(self, x):
        mask = TorchBernoulli(probs=self.prob_reinit).sample((x.size(0),))
        x[mask > 0.5, ...] = self.initialize(int(mask.sum()))
        return x

    def initialize(self, n):
        x = TorchUnif(min(self.range), max(self.range)).sample((n, *self.shape))
        return x


class IndependentReplayBuffer(RingReplayBuffer):
    def __init__(self, data_shape, data_range, maxlen=10000, prob_reinitialize=0.05):
        """
        The other replay buffer ReplayBuffer refreshes elements on a circular cadence;
        This buffer only samples from index i and replaces into index i -- in other
        words, each chain evolves independently.
        """
        super(IndependentReplayBuffer, self).__init__(
            data_shape=data_shape,
            data_range=data_range,
            maxlen=maxlen,
            prob_reinitialize=prob_reinitialize,
        )
        self.latest_ndx = None

    def append(self, new):
        if new.size(0) == self.latest_ndx.numel():
            self.buffer[self.latest_ndx, ...] = new
        else:
            raise ValueError(
                f"cannot append `new` with shape {new.size} using latest_ndx with shape {self.latest_ndx}"
            )

    def sample(self, batch_size):
        assert isinstance(batch_size, int)
        assert batch_size < self.maxlen
        self.latest_ndx = self.rand_index(batch_size)
        out = self.buffer[self.latest_ndx, ...]
        out = self.maybe_reinitialize(out)
        return out

    def rand_index(self, batch_size):
        return torch.randint(self.maxlen, size=(batch_size,))


class EpochIndependentReplayBuffer(IndependentReplayBuffer):
    def __init__(self, data_shape, data_range, maxlen=10000, prob_reinitialize=0.05):
        """
        The ReplayBuffer is a ring buffer. By contrast, this buffer only samples from
        index i and replaces into index i -- in other words,
        each chain evolves independently. Additionally, it samples without replacement
        until the pool is depleted.

        A side-effect to this is that it guarantees that the samples are "stale" as
        the epoch proceeds, creating a sawtooth pattern in the energy for the MCMC
        data. It's not clear that this is truly a downside, though.
        """
        super(IndependentReplayBuffer, self).__init__(
            data_shape=data_shape,
            data_range=data_range,
            maxlen=maxlen,
            prob_reinitialize=prob_reinitialize,
        )
        self.remaining_ndx = np.arange(maxlen)
        self.latest_ndx = None

    def sample(self, batch_size):
        assert isinstance(batch_size, int)
        assert batch_size < self.maxlen
        if self.remaining_ndx.size < batch_size:
            self.remaining_ndx = np.arange(self.maxlen)
        self.latest_ndx = self.rand_index(batch_size)
        out = self.buffer[self.latest_ndx, ...]
        out = self.maybe_reinitialize(out)
        return out

    def rand_index(self, batch_size):
        ndx = np.random.choice(self.remaining_ndx, size=batch_size, replace=False)
        self.remaining_ndx = np.setdiff1d(self.remaining_ndx, ndx, assume_unique=True)
        return torch.LongTensor(ndx)
