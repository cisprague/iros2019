#!/usr/bin/python3
# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import numpy as np, pygmo as pg, matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from scipy.integrate import solve_ivp
from indirect import Indirect

class Pendulum(Indirect):

    def __init__(self, ub=2):

        # initialise indirect trajectory
        Indirect.__init__(self, 4, 1, [3, 2, 2*np.pi, 2])

        self.sf = np.zeros(self.sdim)
        self.beta = 0
        self.ub = ub
        self.bound = True

    def ds(self, s, u):

        # state
        x, v, theta, omega = s

        return np.array([
            v,
            u,
            omega,
            np.sin(theta) - u*np.cos(theta)
        ], float)

    def dds(self, s, u):

        # state
        x, v, theta, omega = s

        return np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, u*np.sin(theta) + np.cos(theta), 0]
        ], float)

    def dsl(self, sl, u):

        # fullstate
        x, v, theta, omega, lx, lv, ltheta, lomega = sl

        # common subexpression elimination
        x0 = np.sin(theta)
        x1 = np.cos(theta)

        # fullstate equations of motion
        return np.array([
            v,
            u,
            omega,
            -u*x1 + x0,
            0,
            -lx,
            -lomega*(u*x0 + x1),
            -ltheta
        ], float)

    def ddsl(self, sl, u):

        # fullstate
        x, v, theta, omega, lx, lv, ltheta, lomega = sl

        # common subexpression elimination
        x0 = np.cos(theta)
        x1 = np.sin(theta)
        x2 = u*x1

        # return 
        return np.array([
            [0, 1,                             0, 0,  0, 0,  0,        0],
            [0, 0,                             0, 0,  0, 0,  0,        0],
            [0, 0,                             0, 1,  0, 0,  0,        0],
            [0, 0,                       x0 + x2, 0,  0, 0,  0,        0],
            [0, 0,                             0, 0,  0, 0,  0,        0],
            [0, 0,                             0, 0, -1, 0,  0,        0],
            [0, 0,           -lomega*(u*x0 - x1), 0,  0, 0,  0, -x0 - x2],
            [0, 0,                             0, 0,  0, 0, -1,        0]
        ], float)

    def u(self, sl, alpha):

        # fullstate
        x, v, theta, omega, lx, lv, ltheta, lomega = sl

        if alpha == 1:
            s = -lomega*np.cos(theta) + lv
            u = -self.ub if s <= 0 else self.ub
        else:
            u = (-lomega*np.cos(theta) + lv)/(2*(alpha - 1))

        if self.bound:
            u = np.clip(u, -self.ub, self.ub)
        return u

    def ubeta(self, sl, alpha):

        # fullstate
        x, v, theta, omega, lx, lv, ltheta, lw = sl
        
        # compute control
        if self.beta == 0:
            u = lw*np.cos(theta)/2 - lv/2
        elif alpha == 0 and self.beta == 0:
            u = np.inf*(-lw*np.cos(theta) + lv + 1)
        elif alpha == 1 and self.beta == 1:
            u = np.inf*(-lw*np.cos(theta) + lv)
        elif self.beta == 1:
            u = np.inf*(-alpha - lw*np.cos(theta) + lv + 1)
        else:
            u = (-alpha*self.beta + self.beta - lw*np.cos(theta) + lv)/(2*(self.beta - 1))

        return np.clip(u, -2, 2)

    def fitness(self, z):

        # duration and costates
        T, l0 = z[0], z[1:]

        # simulate
        tl, sl = self.propagate(T, self.s0, l0, self.alpha, atol=1e-13, rtol=1e-13)

        # state mismatch
        #ec = np.zeros(self.sdim) - sl[-1, :self.sdim]
        ec = self.sf - sl[-1, :self.sdim]
        #ec[0] = sl[-1, 4]
        #ec[2] = np.cos(sl[-1, 2]) - 1

        # fitness vector
        return np.hstack(([0], ec))

    def lagrangian(self, u, alpha, beta):
        return alpha*(beta + (-beta + 1)*abs(u)) + u**2*(-alpha + 1)

    def plot_states(self, tl, sl, ul):

        # get axis
        fig, ax = Indirect.plot_states(self, tl, sl, ul)

        # labels
        labels = [r'$x$', r'$v$', r'$theta$', r'$\omega$', r'$u$']
        
        # apply y labels
        for a, l in zip(ax, labels):
            a.set_ylabel(l)

        # apply x label
        ax[-1].set_xlabel(r'$\tau$')
        

        return fig, ax

    def plot_traj(self, tl, sl):

        # create axis
        fig, ax = plt.subplots(1)

        # compute endpoints
        x = sl[:,0] + np.sin(sl[:,2])
        y = np.cos(sl[:,2])

        # plot endpoints
        ax.plot(x, y, "k.-", alpha=0.1)

        # plot cart
        ax.plot(sl[:,0], np.zeros(len(sl[:,0])), "k.-", alpha=0.1)

        # plot arm
        for i in range(len(x)):
            ax.plot([x[i], sl[i,0]], [y[i], 0], "k.-", alpha=0.1)

        # equal aspect ratio
        ax.set_aspect('equal')

        # set labels
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')

        return fig, ax
        
if __name__ == "__main__":

    # instantiate problem
    seg = Pendulum()

    # initial parameters
    s0 = [0, 0, np.pi, 0]
    alpha = 0
    Tlb = 5
    Tub = 12
    lb = 1

    # solve for initial trajectory
    print("Solving for initial trajectory")
    while True:
        z, f, feas = seg.solve(s0, alpha, Tlb, Tub, lb)
        if feas:
            break
    print(z)

    # forward homotopy
    print("Solving forward homotopy")
    sols = seg.homotopy(s0, alpha, Tlb, Tub, 10, z, 1)
    print(sols)

