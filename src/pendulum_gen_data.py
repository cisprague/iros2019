# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import numpy as np
from pendulum import Pendulum

# instantiate problem
seg = Pendulum()

# trajectory arguments
s0    = [0, 0, np.pi, 0]
alpha = 0
Tlb   = 0
Tub   = 15
lb    = 10

# solve
zg = [10.21795266,  0.19885557,  0.77306339,  0.99997456,  0.83067696]
z, f, feas = seg.solve(s0, alpha, Tlb, Tub, lb, z=zg)

# quadratic database generation
n = 1
nw = 4
T = seg.random_walk_par(s0, z, alpha, n, Tlb, Tub, lb, nw, dsm=0.01, verbose=True)

# database homotopy
res = seg.homotopy_db(T, 0, 0, 15, 10, 1, step=0.01)
np.save('pqhdb.npy', res)
data = seg.gen_db(res, cat=True)
np.save('pqhanndb.npy', data)

from multiprocessing import cpu_count, Pool
from ann import *
import cloudpickle as cp

# load training database
data = np.load('pqhanndb.npy')

# format database
db = Data(data, [0, 1, 2, 3, 4], [5])
#cp.dump(db, open("pendulum_torch_db.p", "wb"))

# neural network architectures
shapes = [
    [5, 50, 50, 1],
    [5, 50, 50, 50, 50, 1],
    [5, 100, 100, 1],
    [5, 100, 100, 100, 100, 1]
]

# training function
def train_mlps(shape):
    mlp = Pendulum_Controller(shape)
    mlp.train(db.i[:1000], db.o[:1000], epo=1000, lr=1e-3, gpu=False, ptst=0.1)
    return mlp

# train
mlps = Pool(cpu_count()).map(train_mlps, shapes)

# save nets
cp.dump(mlps, open("pendulum_nets.p", "wb"))