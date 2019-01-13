#!/usr/bin/python3
# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import numpy as np, pykep as pk
from indirect import Indirect

class Spacecraft(Indirect):

    def __init__(self, T=0.3, isp=2500, m=1000):

        # nondimensionalisation units
        self.L = 149597870691.0
        self.V = 29784.6905
        self.M = m
        self.A = (self.V*self.V)/self.L
        self.F = self.M*self.A
        self.T = self.L/self.V
        self.Q = self.F/self.V

        # constants
        self.c1 = T
        self.c2 = T/(isp*9.80665)
        self.mu = pk.MU_SUN

        # planets
        p0 = pk.planet.jpl_lp('earth')
        pf = pk.planet.jpl_lp('mars')
        
        # keplerian elements
        self.ke0 = np.array(p0.osculating_elements(pk.epoch(0)))
        self.kef = np.array(pf.osculating_elements(pk.epoch(0)))

        # state and control dimensions
        Indirect.__init__(
            self, 7, 4, 
            [self.L]*3 + [self.V]*3 + [self.M]
        )

        # secondary homotopy parameter
        self.beta = 0

    def ds(self, s, u):

        # state 
        x, y, z, vx, vy, vz, m = s

        # control 
        u, ux, uy, uz = u

        # common subexpression elimination
        x0 = self.c1*u/m
        x1 = self.mu/(x**2 + y**2 + z**2)**(3/2.)
        
        # state dyanamics
        return np.array([
            vx,
            vy,
            vz,
            ux*x0 - x1*x,
            uy*x0 - x1*y,
            uz*x0 - x1*z,
            -self.c2*u
        ], float)

    def dds(self, s, u):

        # state
        x, y, z, vx, vy, vz, m = s

        # control
        u, ux, uy, uz = u 
        
        # common subexpression elimination
        x0 = x**2
        x1 = y**2
        x2 = z**2
        x3 = x0 + x1 + x2
        x4 = -self.mu/x3**(3/2.)
        x5 = 3*self.mu/x3**(5/2.)
        x6 = x*x5
        x7 = x6*y
        x8 = x6*z
        x9 = self.c1*u/m**2
        x10 = x5*y*z

        # state dynamics jacobian
        return np.array([
            [         0,          0,          0, 1, 0, 0,             0],
            [         0,          0,          0, 0, 1, 0,             0],
            [         0,          0,          0, 0, 0, 1,             0],
            [x0*x5 + x4,         x7,         x8, 0, 0, 0, -ux*x9],
            [        x7, x1*x5 + x4,        x10, 0, 0, 0, -uy*x9],
            [        x8,        x10, x2*x5 + x4, 0, 0, 0, -uz*x9],
            [         0,          0,          0, 0, 0, 0,             0]
        ], float)

    def dsl(self, sl, u):

        # state
        x, y, z, vx, vy, vz, m, lx, ly, lz, lvx, lvy, lvz, lm = sl

        # control
        u, ux, uy, uz = u
        
        # common subexpression elimination
        x0 = self.c1*u
        x1 = x0/m
        x2 = x**2
        x3 = y**2
        x4 = z**2
        x5 = x2 + x3 + x4
        x6 = self.mu/x5**(3/2.)
        x7 = 3*self.mu/x5**(5/2.)
        x8 = x*x7
        x9 = x8*y
        x10 = lvz*z
        x11 = -x6
        x12 = x7*y
        x13 = x0/m**2
        
        # fullstate dynamics
        return np.array([
            vx,
            vy,
            vz,
            ux*x1 - x*x6,
            uy*x1 - x6*y,
            uz*x1 - x6*z,
            -self.c2*u,
            -lvx*(x11 + x2*x7) - lvy*x9 - x10*x8,
            -lvx*x9 - lvy*(x11 + x3*x7) - x10*x12,
            -lvx*x8*z - lvy*x12*z - lvz*(x11 + x4*x7),
            -lx,
            -ly,
            -lz,
            ux*lvx*x13 + uy*lvy*x13 + uz*lvz*x13
        ], float)

    def ddsl(self, sl, u):

        # state
        x, y, z, vx, vy, vz, m, lx, ly, lz, lvx, lvy, lvz, lm = sl

        # control
        u, ux, uy, uz = u
        
        # common subexpression elimination
        x0 = x**2
        x1 = y**2
        x2 = z**2
        x3 = x0 + x1 + x2
        x4 = self.mu/x3**(3/2.)
        x5 = -x4
        x6 = self.mu/x3**(5/2.)
        x7 = 3*x6
        x8 = x0*x7
        x9 = x7*y
        x10 = x*x9
        x11 = x7*z
        x12 = x*x11
        x13 = self.c1*u
        x14 = x13/m**2
        x15 = ux*x14
        x16 = x1*x7
        x17 = x9*z
        x18 = uy*x14
        x19 = x2*x7
        x20 = uz*x14
        x21 = -lvy*x9
        x22 = -lvz*x11
        x23 = 15*self.mu/x3**(7/2.)
        x24 = x0*x23
        x25 = x24*y
        x26 = x24*z
        x27 = 9*x6
        x28 = x*x7
        x29 = x*x23*y*z
        x30 = lvz*x29
        x31 = x1*x23
        x32 = x*x31
        x33 = lvy*x29
        x34 = x2*x23
        x35 = x*x34
        x36 = -x10
        x37 = -x12
        x38 = -lvx*x28
        x39 = x31*z
        x40 = lvx*x29
        x41 = x34*y
        x42 = -x17
        x43 = 2*x13/m**3
        
        # fullstate jacobian
        return np.array([
            [                                                                                     0,                                                                                     0,                                                                                     0, 1, 0, 0,                                                                                        0,  0,  0,  0,       0,         0,         0, 0],
            [                                                                                     0,                                                                                     0,                                                                                     0, 0, 1, 0,                                                                                        0,  0,  0,  0,       0,         0,         0, 0],
            [                                                                                     0,                                                                                     0,                                                                                     0, 0, 0, 1,                                                                                        0,  0,  0,  0,       0,         0,         0, 0],
            [                                                                               x5 + x8,                                                                                   x10,                                                                                   x12, 0, 0, 0,                                                                                     -x15,  0,  0,  0,       0,         0,         0, 0],
            [                                                                                   x10,                                                                              x16 + x5,                                                                                   x17, 0, 0, 0,                                                                                     -x18,  0,  0,  0,       0,         0,         0, 0],
            [                                                                                   x12,                                                                                   x17,                                                                              x19 + x5, 0, 0, 0,                                                                                     -x20,  0,  0,  0,       0,         0,         0, 0],
            [                                                                                     0,                                                                                     0,                                                                                     0, 0, 0, 0,                                                                                        0,  0,  0,  0,       0,         0,         0, 0],
            [-lvx*(-x**3*x23 + x*x27) + lvy*x25 + lvz*x26 + x21 + x22,              -lvx*(-x25 + x9) - lvy*x28 + lvy*x32 + x30,              -lvx*(x11 - x26) - lvz*x28 + lvz*x35 + x33, 0, 0, 0,                                                                                        0,  0,  0,  0, x4 - x8,       x36,       x37, 0],
            [                lvx*x25 - lvx*x9 - lvy*(x28 - x32) + x30, lvx*x32 - lvy*(-x23*y**3 + x27*y) + lvz*x39 + x22 + x38,               -lvy*(x11 - x39) + lvz*x41 - lvz*x9 + x40, 0, 0, 0,                                                                                        0,  0,  0,  0,     x36, -x16 + x4,       x42, 0],
            [              -lvx*x11 + lvx*x26 - lvz*(x28 - x35) + x33,              -lvy*x11 + lvy*x39 - lvz*(-x41 + x9) + x40, lvx*x35 + lvy*x41 - lvz*(-x23*z**3 + x27*z) + x21 + x38, 0, 0, 0,                                                                                        0,  0,  0,  0,     x37,       x42, -x19 + x4, 0],
            [                                                                                     0,                                                                                     0,                                                                                     0, 0, 0, 0,                                                                                        0, -1,  0,  0,       0,         0,         0, 0],
            [                                                                                     0,                                                                                     0,                                                                                     0, 0, 0, 0,                                                                                        0,  0, -1,  0,       0,         0,         0, 0],
            [                                                                                     0,                                                                                     0,                                                                                     0, 0, 0, 0,                                                                                        0,  0,  0, -1,       0,         0,         0, 0],
            [                                                                                     0,                                                                                     0,                                                                                     0, 0, 0, 0, -ux*lvx*x43 - uy*lvy*x43 - uz*lvz*x43,  0,  0,  0,     x15,       x18,       x20, 0]
        ], float)

    def u(self, sl, alpha):

        # fullstate
        x, y, z, vx, vy, vz, m, lx, ly, lz, lvx, lvy, lvz, lm = sl

        # common subexpression elimination
        x0 = lvx**2
        x1 = lvy**2
        x2 = lvz**2
        x3 = np.sqrt(x0 + x1 + x2)
        x4 = m*x3
        x5 = -lm*self.c2*x4 - self.c1*x0 - self.c1*x1 - self.c1*x2
        x6 = 1/(m*x3)
        x7 = x5*x6
        x8 = x7/2
        x9 = -x8
        x10 = x4 + x5
        x11 = np.inf*x6
        x12 = self.beta*x4
        x13 = x12 + x5
        x14 = 1/(self.beta - 1)
        x15 = x14*x6/2.
        
        # control throttles
        if alpha == 0 and self.beta == 0:
            u = x9
        elif alpha == 0 and self.beta == 1:
            u = x10*x11
        elif alpha == 1 and self.beta == 0:
            u = x9
        elif alpha == 1 and self.beta == 1:
            u = np.inf*x7
        elif alpha == 0:
            u = x13*x15
        elif alpha == 1:
            u = x14*x8
        elif self.beta == 0:
            u = x9
        elif self.beta == 1:
            u = x11*(-alpha*x4 + x10)
        else:
            u = x15*(-alpha*x12 + x13)

        # bound control
        u = np.clip(u, 0, 1)
        
        # thrust direction
        ux = -lvx/np.sqrt(lvx**2 + lvy**2 + lvz**2)
        uy = -lvy/np.sqrt(lvx**2 + lvy**2 + lvz**2)
        uz = -lvz/np.sqrt(lvx**2 + lvy**2 + lvz**2)

        # return full control
        return np.array([u, ux, uy, uz])

    def propagate(self, T, s0, l0, alpha, atol=1e-12, rtol=1e-12, u=False):

        # nondimensionalise initial state
        s0 = np.copy(s0)
        s0[0:3] /= self.L 
        s0[3:6] /= self.V
        s0[6]   /= self.M

        # duration from mjd2000 to mjs2000
        T = pk.epoch(T).mjd2000*24*60*60

        # nondimensionalise time
        T /= self.T

        # nondimensionalise constants
        self.c1 /= self.F
        self.c2 /= self.Q
        self.mu /= pk.MU_SUN

        # propagate
        if u:
            tl, sl, ul = Indirect.propagate(self, T, s0, l0, alpha, atol=atol, rtol=rtol, u=True)
        else:
            tl, sl = Indirect.propagate(self, T, s0, l0, alpha, atol=atol, rtol=rtol, u=False)

        # redimensionalise states
        sl[:, 0:3] *= self.L
        sl[:, 3:6] *= self.V
        sl[:, 6]   *= self.M

        # redimensionalise times
        tl *= self.T
        tl /= 24*60*60

        # redimensionalise constants
        self.c1 *= self.F
        self.c2 *= self.Q
        self.mu *= pk.MU_SUN

        # return times, states, and possibly controls
        if u:
            return tl, sl, ul
        else:
            return tl, sl

    def fitness(self, z):

        # duration
        T = z[0]

        # final eccentric annomolies
        Mf = z[1]

        # initial costates
        l0 = z[2:]

        # set final eccentric anomoly
        self.kef[5] = Mf

        # final position and velocity
        rf, vf = pk.par2ic(self.kef, pk.MU_SUN)
        rf = np.array(rf)
        vf = np.array(vf)

        # propagate trajectory
        tl, sl = self.propagate(T, self.s0, l0, self.alpha, atol=1e-10, rtol=1e-10)

        # compute position and velocity
        dp = (sl[-1, 0:3] - rf)/self.L
        dv = (sl[-1, 3:6] - vf)/self.V

        # mass transversality
        lmf = sl[-1, 13]

        # final eccentric anomoly transversality
        lambdasf = sl[-1, 7:13]
        rfnorm = np.sqrt(rf[0] * rf[0] + rf[1] * rf[1] + rf[2] * rf[2])
        tmp = - pk.MU_SUN / rfnorm**3
        tangent = np.array([vf[0], vf[1], vf[2], tmp *
                            rf[0], tmp * rf[1], tmp * rf[2]])
        tangent_norm = np.linalg.norm(tangent)
        tangent = tangent / tangent_norm
        Tf = np.dot(lambdasf, tangent)

        # return equality constraints
        return np.array([0, *dp, *dv, lmf, Tf])

    def get_nec(self):
        return self.sdim + 1

    def get_bounds(self):
        lb = [self.Tlb, -4 * np.pi] + [-self.lb] * self.sdim
        ub = [self.Tub, 4 * np.pi] + [self.lb] * self.sdim
        return lb, ub

