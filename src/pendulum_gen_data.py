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
n = 5
nw = 10
T = seg.random_walk_par(s0, z, alpha, n, Tlb, Tub, lb, nw, dsm=0.01, verbose=True)
np.save("pendulum_rw.npy", T)

# database homotopy
res = seg.homotopy_db(T, 0, 0, 15, 10, 1, step=0.01)
np.save('pendulum_hdb.npy', res)
data = seg.gen_db(res, cat=True)
np.save('pqhanndb.npy', data)