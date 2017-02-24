from __future__ import absolute_import, division

import autograd.numpy as np
import autograd.numpy.random as npr

from autograd.scipy.misc import logsumexp
import autograd.scipy.signal
from autograd.util import flatten, flatten_func
from autograd import elementwise_grad, grad, hessian

import os, struct, array, pdb, time, gc, argparse
import cPickle as pickle

parser = argparse.ArgumentParser(description='Hessian of MNIST (conv)')
parser.add_argument('-s', '--seed',      help='Random seed',            type=int, default=42)
parser.add_argument('-o', '--output',    help='Save eigenvalues here',   type=str, required=True)
parser.add_argument('--hessian_num_batches',    help='Hessian batches',   type=int, default = 128)
parser.add_argument('--save_hessian',       help='Save Hessian',   action='store_true')
parser.add_argument('--max_epochs',    help='Max. epochs',   type=int, default = 50)
args = vars(parser.parse_args())
dtype = np.float32

convolve = autograd.scipy.signal.convolve

opt = {
    'batch_size' : 32,
    'lr' : 0.001,
    'lrd' : 0.98,
    'scale' : 1e-1,
    'max_epochs' : args['max_epochs'],
    'full' : True,
    'cnn' : True,
    'width': 28
}
opt.update(args)
#print opt

def bin_ndarray(ndarray, new_shape, operation='mean'):
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation {} not supported.".format(operation))
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray

def load_mnist(dir, opt, dtype = np.float32):
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:,None] == np.arange(k)[None, :], dtype=int)

    def parse_labels(fp):
        with open(dir + fp, 'rb') as fh:
            _, n = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(fp):
        with open(dir + fp, 'rb') as fh:
            _, n, r, c = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(n, r, c)

    tx = parse_images('/train-images-idx3-ubyte')
    ty = parse_labels('/train-labels-idx1-ubyte')
    vx  = parse_images('/t10k-images-idx3-ubyte')
    vy  = parse_labels('/t10k-labels-idx1-ubyte')

    w = opt['width']
    tx = bin_ndarray(tx, (tx.shape[0], w,w))
    vx = bin_ndarray(vx, (vx.shape[0], w,w))
    tx = tx.reshape(tx.shape[0], 1, w,w)
    vx = vx.reshape(vx.shape[0], 1, w,w)
    tx /= 255.0
    vx /= 255.0

    if not opt['cnn']:
        tx = partial_flatten(tx)
        vx  = partial_flatten(vx)

    ty = one_hot(ty, 10)
    vy = one_hot(vy, 10)

    frac = (opt['full'] and 1) or 0.01
    tn, vn = int(tx.shape[0]*frac), int(vx.shape[0]*frac)

    idx = np.random.permutation(range(tx.shape[0]))[:tn]
    tx, ty = tx[idx], ty[idx]

    idx = np.random.permutation(range(vx.shape[0]))[:vn]
    vx, vy = vx[idx], vy[idx]

    ret = (tx.shape[0], tx.astype(dtype), ty.astype(dtype), vx.astype(dtype), vy.astype(dtype))
    #pickle.dump(ret, open('mnist.pkl', 'wb'))
    return ret

def sgd(grad, init_params, callback=None, num_iters=200, step_size=0.1, mass=0.9):
    """Stochastic gradient descent with momentum.
    grad() must have signature grad(x, i), where i is the iteration number."""
    flattened_grad, unflatten, x = flatten_func(grad, init_params)

    velocity = np.zeros(len(x))
    for i in range(num_iters):
        g = flattened_grad(x, i)
        if callback: callback(unflatten(x), i, unflatten(g))
        velocity = mass * velocity - (1.0 - mass) * g
        x = x + step_size * velocity
    return unflatten(x)

def init_params(scale, rs = npr.RandomState(0)):
    w = range(4)

    # LeNet:    20-50-500
    #           10-20-(320)-128-10
    w[0] = (scale*rs.randn(1,10, 5,5).astype(dtype),
            scale*rs.randn(1,10,1,1).astype(dtype))
    w[1] = (scale*rs.randn(10,20, 5,5).astype(dtype),
            scale*rs.randn(1,20,1,1).astype(dtype))
    w[2] = (scale*rs.randn(320, 128).astype(dtype),
            scale*rs.randn(128).astype(dtype))
    w[3] = (scale*rs.randn(128, 10).astype(dtype),
            scale*rs.randn(10).astype(dtype))

    t1,_ = flatten(w)
    print '[size]: ', t1.shape
    return w

def maxpool(x, k):
    newsz = x.shape[:2]
    sz = x.shape[2:]
    newsz += (k[0], sz[0]//k[0])
    newsz += (k[1], sz[1]//k[1])
    r = x.reshape(newsz)
    return np.max(np.max(r, axis=2), axis=3)

def predict(p, x):
    relu = lambda _x: np.maximum(_x, 0.)

    x = relu(convolve(x, p[0][0], axes=([2,3],[2,3]), dot_axes=([1], [0]), mode='valid') + p[0][1])
    x = maxpool(x, (2,2))
    x = relu(convolve(x, p[1][0], axes=([2,3],[2,3]), dot_axes=([1], [0]), mode='valid') + p[1][1])
    x = maxpool(x, (2,2))
    x = x.reshape(x.shape[0], -1)

    for w, b in p[2:]:
        yh = np.dot(x, w) + b
        x = relu(yh)
    return yh - logsumexp(yh, axis=1, keepdims=True)

def log_posterior(p, x, y):
    return np.sum(predict(p, x)*y)

def accuracy(p, x, y):
    c = np.argmax(y, axis=1)
    ch = np.argmax(predict(p, x), axis=1)
    return np.mean(c == ch)*100

def objective(p, i):
    idx = i % opt['num_batches']
    b = slice(idx*opt['batch_size'], (idx+1)*opt['batch_size'])
    if i % 10 == 0:
        print i, accuracy(p, tx[b], ty[b])
    return -log_posterior(p, tx[b], ty[b])
objective_grad = grad(objective)

n, tx, ty, vx, vy = load_mnist('../proc/raw', opt)
p = init_params(opt['scale'], npr.RandomState(opt['seed']))
opt['num_batches'] = int(np.ceil(n / opt['batch_size']))

s = time.time()
for e in xrange(opt['max_epochs']):
    lr = opt['lr']*opt['lrd']**e

    def stats(p, i, g):
        if i % opt['num_batches'] == 0:
            te = accuracy(p, tx, ty)
            ve = accuracy(p, vx, vy)
            print '{:15}|{:20}|{:20}|'.format(e, te, ve)
        if i % 10 == 0:
            print ('[%03d][%03d/%03d]')%(e, i%opt['num_batches'], opt['num_batches'])
        gc.collect()

    p = sgd(objective_grad, p,
                step_size = lr, num_iters=opt['num_batches'],
                callback=stats)
print '[opt] ', time.time()-s
params = p

print '[flat params] ...'
flat_f, unflatten, flat_params = flatten_func(objective, params)

print '[flat hess] ...'
flat_hess = hessian(flat_f)

h = None
print '[compute hess] ...'
for i in np.random.permutation(np.arange(opt['num_batches']))[:opt['hessian_num_batches']]:
    if h is None:
        h = flat_hess(flat_params, i)
    else:
        np.add(h, flat_hess(flat_params, i), h)
    print '[progress] ', i, ' dt: ', time.time()-s
    gc.collect()
h = h.squeeze()/float(opt['hessian_num_batches']*opt['batch_size'])
print '[hessian] ', time.time() -s

if opt['save_hessian']:
    np.save(opt['save_hessian']+'.hes', h)

e = np.linalg.eigvals(h)
np.save(opt['output']+'.eig', e)
