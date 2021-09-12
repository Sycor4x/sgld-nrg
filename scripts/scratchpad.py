#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2021-09-11 (year-month-day)

from sgld_nrg.sgld import ReplayBuffer, SgldLogitEnergy
from sgld_nrg.networks import SimpleNet

if __name__ == "__main__":
    my_net = SimpleNet()
    my_buff = ReplayBuffer(data_shape=(1, 28, 28), data_range=(-1.0, 1.0), maxlen=3)
    my_sgld = SgldLogitEnergy(net=my_net, replay_buffer=my_buff)
    foo = my_sgld(2)
    print(foo[0, ...])
