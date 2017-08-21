from __future__ import print_function
import argparse, math, random
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from timeit import default_timer as timer
import torch.backends.cudnn as cudnn

import models, loader, optim
import numpy as np
from utils import *

parser = argparse.ArgumentParser(description='PyTorch Entropy-SGD')
ap = parser.add_argument
ap('-m', help='mnistfc | mnistconv | allcnn', type=str, default='mnistconv')
ap('-b',help='Batch size', type=int, default=128)
ap('-B', help='Max epochs', type=int, default=100)
ap('--lr', help='Learning rate', type=float, default=0.1)
ap('--l2', help='L2', type=float, default=0.0)
ap('-L', help='Langevin iterations', type=int, default=0)
ap('--gamma', help='gamma', type=float, default=1e-4)
ap('--scoping', help='scoping', type=float, default=1e-3)
ap('--noise', help='SGLD noise', type=float, default=1e-4)
ap('-g', help='GPU idx.', type=int, default=1)
ap('--no_cuda', help='run on gpu', action='store_true')
ap('-s', help='seed', type=int, default=42)
opt = vars(parser.parse_args())

th.set_num_threads(2)
if opt['no_cuda']:
    opt['g'] = -1
    th.cuda.set_device(opt['g'])
    th.cuda.manual_seed(opt['s'])
    cudnn.benchmark = True
random.seed(opt['s'])
np.random.seed(opt['s'])
th.manual_seed(opt['s'])

if 'mnist' in opt['m']:
    opt['dataset'] = 'mnist'
elif 'allcnn' in opt['m']:
    opt['dataset'] = 'cifar10'
else:
    assert False, "Unknown opt['m']: " + opt['m']

train_loader, val_loader, test_loader = getattr(loader, opt['dataset'])(opt)
model = getattr(models, opt['m'])(opt)
criterion = nn.CrossEntropyLoss()
if not opt['no_cuda']:
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = optim.EntropySGD(model.parameters(),
        config = dict(lr=opt['lr'], momentum=0.9, nesterov=True, weight_decay=opt['l2'],
        L=opt['L'], eps=opt['noise'], g0=opt['gamma'], g1=opt['scoping']))

print(opt)

def train(e):
    model.train()

    fs, top1 = AverageMeter(), AverageMeter()
    ts = timer()

    bsz = opt['b']
    maxb = int(math.ceil(train_loader.n/bsz))

    for bi in xrange(maxb):
        def helper():
            def feval():
                x,y = next(train_loader)
                if not opt['no_cuda']:
                    x,y = x.cuda(), y.cuda()

                x, y = Variable(x), Variable(y.squeeze())
                bsz = x.size(0)

                optimizer.zero_grad()
                yh = model(x)
                f = criterion.forward(yh, y)
                f.backward()

                prec1, = accuracy(yh.data, y.data, topk=(1,))
                err = 100.-prec1[0]
                return (f.data[0], err)
            return feval

        f, err = optimizer.step(helper(), model, criterion)

        fs.update(f, bsz)
        top1.update(err, bsz)

        if bi % 100 == 0 and bi != 0:
            print('[%2d][%4d/%4d] %2.4f %2.2f%%'%(e,bi,maxb, fs.avg, top1.avg))

    print('Train: [%2d] %2.4f %2.2f%% [%.2fs]'% (e, fs.avg, top1.avg, timer()-ts))
    print()

def set_dropout(cache = None, p=0):
    if cache is None:
        cache = []
        for l in model.modules():
            if 'Dropout' in str(type(l)):
                cache.append(l.p)
                l.p = p
        return cache
    else:
        for l in model.modules():
            if 'Dropout' in str(type(l)):
                assert len(cache) > 0, 'cache is empty'
                l.p = cache.pop(0)

def dry_feed():
    cache = set_dropout()
    maxb = int(math.ceil(train_loader.n/opt['b']))
    for bi in xrange(maxb):
        x,y = next(train_loader)
        if not opt['no_cuda']:
            x,y = x.cuda(), y.cuda()
        x,y =   Variable(x, volatile=True), \
                Variable(y.squeeze(), volatile=True)
        yh = model(x)
    set_dropout(cache)

def val(e, data_loader):
    dry_feed()
    model.eval()

    maxb = int(math.ceil(data_loader.n/opt['b']))

    fs, top1 = AverageMeter(), AverageMeter()
    for bi in xrange(maxb):
        x,y = next(data_loader)
        bsz = x.size(0)

        if not opt['no_cuda']:
            x,y = x.cuda(), y.cuda()

        x,y =   Variable(x, volatile=True), \
                Variable(y.squeeze(), volatile=True)
        yh = model(x)

        f = criterion.forward(yh, y).data[0]
        prec1, = accuracy(yh.data, y.data, topk=(1,))
        err = 100-prec1[0]

        fs.update(f, bsz)
        top1.update(err, bsz)

    print('Test: [%2d] %2.4f %2.4f%%\n'%(e, fs.avg, top1.avg))
    print()

for e in xrange(opt['B']):
    train(e)
    if e % 5 == 0:
        val(e, val_loader)
