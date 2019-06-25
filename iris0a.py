import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


X = np.loadtxt('iris/iris-x.txt').astype(np.float)
Y = np.loadtxt('iris/iris-y.txt')

class IrisChain(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
            l1 = L.Linear(4, 6),
            l2 = L.Linear(6, 3),
        )
    def __call__(self, x, y):
        return F.mean_squared_error(self.fwd(x), y)