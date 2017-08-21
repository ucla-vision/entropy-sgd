## Entropy-SGD: Biasing Gradient Descent Into Wide Valleys

This is the implementation for [Entropy-SGD: Biasing Gradient Descent Into Wide Valleys](https://arxiv.org/abs/1611.01838) which will be presented at [ICLR '17](http://iclr.cc). It contains a Lua implementation which was used for the experiments in the paper as well as an identical implementation in PyTorch in the [python](python) folder.

-----------------------------

### Instructions for Lua

1. You will need Torch installed with CUDNN. We have set up the code for training on MNIST and CIFAR-10 datasets. For the former, we use the mnist package which can be installed by ``luarocks install mnist``. To train on CIFAR-10, download the dataset from [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz). The script [process_cifar.py](process_cifar.py) performs the standard ZCA whitening of the dataset which was used for our experiments.

2. Please run ``th train.lua -h`` to check out command line options. The code is set up for using vanilla SGD on LeNet and should converge to an error of about 0.5% after 100 epochs. To run the same network using Entropy-SGD, you can execute
   ```
   th train.lua -m mnistconv -L 20 --gamma 1e-4 --scoping 1e-3 --noise 1e-4
   ```
The parameters perform the following functions:
    - L is the number of Langevin updates in the algorithm, this controls the exploration and is usually set to 20 in our experiments;
    - gamma is also called "scope" in the paper and controls the forcing term that prevents the Langevin updates from exploring too far;
    - scoping is a technique that progressively increases the value of gamma during the course of training, this modulates gamma as a function of time: ``gamma (1 + scoping)^t`` where ``t`` is the number of parameter updates;
    - noise is the temperature term in Langevin dynamics.

3. Note that running the code with ``-L 0`` argument (default) will use vanilla SGD with Nesterov's momentum. We also collect some run-time statistics of Entropy-SGD such as the gradient norms, direction of the local entropy gradient vs. original stochastic gradient etc. You can see these using the ``-v / --verbose`` option.

4. For CIFAR-10, run
   ```
   th train.lua -m cifarconv --L2 1e-3
   th train.lua -m cifarconv -L 20 --gamma 0.03 --L2 1e-3
   ```
for SGD or Entropy-SGD respectively.

-----------------------------

### Instructions for PyTorch

The code for this is inside the [python](python) folder. You will need the Python packages `torch` and `torchvision` installed from [pytorch.org](pytorch.org).

1. The MNIST example downloads and processes the dataset the first time it is run. The files will be stored in the `proc` folder (same as CIFAR-10 in the Lua version)

2. Run ``python train.py -h`` to check out the command line arguments. The default is to run SGD with Nesterov's momentum on LeNet. You can run Entropy-SGD with
   ```
   python train.py -m mnistconv -L 20 --gamma 1e-4 --scoping 1e-3 --noise 1e-4
   ```
Everything else is identical to the Lua version.

-----------------------------

### Computing the Hessian

The code in [hessian.py](python/hessian.py) computes the Hessian for a small convolutional neural network using SGD and Autograd. Please note that this takes a lot of time, a day or so, and you need to be careful of the memory usage. The experiments in the paper were run on EC2 with 256 GB RAM. Note that this code uses the MNIST dataset downloaded when you run the PyTorch step above.
