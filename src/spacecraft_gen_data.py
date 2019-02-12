# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import numpy as np, pykep as pk
from spacecraft import Spacecraft

# instantiate problem
seg = Spacecraft(T=0.2, isp=2500, m=1000)

# trajectory arguments
s0 = np.hstack((*pk.planet.jpl_lp('earth').eph(pk.epoch(0)), seg.M))
alpha = 0
seg.beta = 0
Tlb = 100
Tub = 500
lb = 100

# solve
print("Solving nominal trajectory...")
zg = [379.20912013,   0.93398202,  12.76567896, -45.55493263,  -8.26940351, 49.99989562, 3.00127225, -8.58992837, 8.32972382]
z, f, feas = seg.solve(s0, alpha, Tlb, Tub, lb, z=zg)

# random walks (alpha=0, beta=0)
print("Generating quadratic database...")
n = 5
nw = 4
T = seg.random_walk_par(s0, z, alpha, n, Tlb, Tub, lb, nw, dsm=0.01, verbose=True)

# database homtopy in beta
print("Performing homotopy in beta...")
beta = 0
betag = 0.999
res = seg.homotopy_db_beta(T, alpha, beta, Tlb, Tub, lb, betag, step=0.1)

# database homotopy in alpha
print("Performing homotopy in alpha...")
s0zl = res[np.argwhere(res[:,2] >= 0.998).flatten(), :2]
alpha = 0
alphag = 1
seg.beta == 0.99999
T = seg.homotopy_db(s0zl, alpha, Tlb, Tub, lb, alphag, step=0.01)
np.save('spacecraft_db_alpha_homotopy_beta_1.npy', T)
db = seg.gen_db(T, cat=True)
np.save('spacecraft_db.npy', db)