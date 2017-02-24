import torch as th
from torchvision import datasets, transforms
import torch.utils.data
import numpy as np
import os, sys, pdb

class sampler_t:
    def __init__(self, batch_size, x,y, train=True):
        self.n = x.size(0)
        self.x, self.y = x,y
        self.b = batch_size
        self.idx = th.range(0, self.b-1).long()
        self.train = train
        self.sidx = 0

    def __next__(self):
        if self.train:
            self.idx.random_(0,self.n-1)
        else:
            s = self.sidx
            e = min(s+self.b-1, self.n-1)
            #print s,e

            self.idx = th.range(s, e).long()
            self.sidx += self.b
            if self.sidx >= self.n:
                self.sidx = 0

        x,y  = th.index_select(self.x, 0, self.idx), \
            th.index_select(self.y, 0, self.idx)
        return x, y

    next = __next__

    def __iter__(self):
        return self

def mnist(opt):
    d1, d2 = datasets.MNIST('../proc', download=True, train=True), \
            datasets.MNIST('../proc', train=False)
    # d1.train_data = (d1.train_data.float() - 126.)/126.
    # d2.test_data = (d2.test_data.float() - 126.)/126.

    train = sampler_t(opt['b'], d1.train_data.view(-1,1,28,28).float(),
        d1.train_labels)
    val = sampler_t(opt['b'], d2.test_data.view(-1,1,28,28).float(),
        d2.test_labels, train=False)
    return train, val, val

def cifar10(opt):
    loc = '../proc/'
    d1 = np.load(loc+'cifar10-train.npz')
    d2 = np.load(loc+'cifar10-test.npz')

    train = sampler_t(opt['b'], th.from_numpy(d1['data']),
                     th.from_numpy(d1['labels']))
    val = sampler_t(opt['b'], th.from_numpy(d2['data']),
                     th.from_numpy(d2['labels']), train=False)
    return train, val, val