#!/usr/bin/python3
# Christopher Iliffe Sprague
# christopher.iliffe.sprague

import numpy as np, pygmo as pg, matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from scipy.integrate import solve_ivp
from multiprocessing import Pool, cpu_count

class Indirect(object):

    def __init__(self, sdim, udim, sbound):

        # state and control dimensions
        self.sdim = int(sdim)
        self.udim = int(udim)

        # state scales
        self.sbound = np.array(sbound, float)


    def propagate(self, T, s0, l0, alpha, atol=1e-12, rtol=1e-12, u=False):

        # numerically integrate
        sol = solve_ivp(
            lambda t, sl: self.dsl(
                sl, 
                self.u(sl, alpha)
            ),
            (0, T),
            np.hstack((s0, l0)),
            method='LSODA',
            atol=atol,
            rtol=rtol,
            jac = lambda t, sl: self.ddsl(
                sl,
                self.u(sl, alpha)
            )
        )

        # time and states
        tl, sl = sol.t, sol.y.T

        # if control trajectory is requested
        if u:
            ul = np.apply_along_axis(
                lambda sl: self.u(sl, alpha),
                1,
                sl
            )
            return tl, sl, ul
        else:
            return tl, sl
    
    def propagate_controller(self, T, s0, u, alpha, atol=1e-12, rtol=1e-12):

        # numerically integrate
        sol = solve_ivp(
            lambda t, s: self.ds(
                s, 
                u(s, alpha)
            ),
            (0, T),
            s0,
            method='LSODA',
            atol=atol,
            rtol=rtol,
            jac = lambda t, s: self.dds(
                s,
                u(s, alpha)
            )
        )

        # time and states
        tl, sl = sol.t, sol.y.T

        # if control trajectory is requested
        ul = np.apply_along_axis(
            lambda sa: u(sa, alpha),
            1,
            sl
        )
        return tl, sl, ul

    def solve_par(self, s0, alpha, Tlb, Tub, lb, npar=cpu_count(), neval=200):

        # intial state
        self.s0    = np.array(s0, float)

        # homotopy parameter
        self.alpha = float(alpha)

        # duration bounds
        self.Tlb   = float(Tlb)
        self.Tub   = float(Tub)

        # costate magnitude bound
        self.lb    = float(lb)

        # problem
        prob = pg.problem(self)
        prob.c_tol = 1e-5

        # algorithm
        algo = pg.nlopt(solver="slsqp")
        algo.maxeval = neval
        algo.xtol_rel = 1e-7
        algo.xtol_abs = 1e-7
        algo = pg.algorithm(algo)
        algo.set_verbosity(5)

        # archipelago
        archi = pg.archipelago(npar, algo=algo, prob=prob, pop_size=1)

        # optimise 
        archi.evolve()
        archi.wait()

        # solutions
        z = archi.get_champions_x()
        f = archi.get_champions_f()
        feas = [prob.feasibility_x(x) for x in z]
        
        return z, f, feas

    def solve(self, s0, alpha, Tlb, Tub, lb, z=None, neval=200):

        # intial state
        self.s0    = np.array(s0, float)

        # homotopy parameter
        self.alpha = float(alpha)

        # duration bounds
        self.Tlb   = float(Tlb)
        self.Tub   = float(Tub)

        # costate magnitude bound
        self.lb    = float(lb)
        
        # problem
        prob = pg.problem(self)
        prob.c_tol = 1e-8

        # algorithm
        algo = pg.nlopt(solver="slsqp")
        algo.maxeval = neval
        algo.xtol_rel = 1e-10
        algo.xtol_abs = 1e-10
        algo = pg.algorithm(algo)
        algo.set_verbosity(5)
        
        # supplied guess
        if z is None:
            pop = pg.population(prob, 1)

        # random guess
        else:
            pop = pg.population(prob, 0)
            pop.push_back(z)

        # solve
        pop = algo.evolve(pop)
        return pop.champion_x, pop.champion_f, prob.feasibility_x(pop.champion_x)

    def get_bounds(self):
        lb = [self.Tlb] + [-self.lb]*self.sdim
        ub = [self.Tub] + [self.lb]*self.sdim 
        return lb, ub

    def get_nobj(self):
        return 1

    def get_nec(self):
        return self.sdim

    def gradient(self, z):
        return pg.estimate_gradient(self.fitness, z)

    def homotopy(self, s0, alpha, Tlb, Tub, lb, z, alphag, step=0.01, verbose=False):

        # step direction
        step = step if alphag - alpha > 0 else -step

        # solve problem with initial decision vector
        zo, f, feas = self.solve(s0, alpha, Tlb, Tub, lb, z)

        # continue only if intial configuration is succesfull
        if feas:
            pass
        else:
            return None

        # solution record
        sols = list()

        # homotopy loop
        i = 0
        while i < 2:

            # solve trajectory
            z, f, feas = self.solve(s0, alpha, Tlb, Tub, lb, z=zo)

            # feasible
            if feas:

                # progress message
                print("z={}\na={}".format(z, self.alpha))

                # store solution
                zo = z
                alphao = alpha

                # record solution
                sols.append((zo, alphao))

                # finished homotopy
                if alpha == alphag:
                    print("Finished")
                    break

                # attempting homotopy boundary
                elif (step < 0 and alpha < 0.001) or (step > 0 and alpha > 0.99):
                    alpha = 0 if step < 0 else 1
                    i += 1

                # change homotopy parameter
                else:

                    # bounded step
                    bstep = min(abs(alpha - (alphag + alpha)/2), abs(step))

                    # apply step 
                    alpha += bstep if step > 0 else -bstep

            # infeasible
            else:
                alpha = (alphao + alpha)/2

        # return solutions
        return np.array(sols)

    def homotopy_beta(self, s0, alpha, beta, Tlb, Tub, lb, z, betag, step=0.01, verbose=False):

            # step direction
            step = step if betag - beta > 0 else -step

            # solve problem with initial decision vector
            self.beta = beta
            betas = beta
            zo, f, feas = self.solve(s0, alpha, Tlb, Tub, lb, z)

            # continue only if intial configuration is succesfull
            if feas:
                pass
            else:
                return None

            # solution record
            sols = list()

            # homotopy loop
            i = 0
            while i < 2:

                # solve trajectory
                z, f, feas = self.solve(s0, alpha, Tlb, Tub, lb, z=zo)

                # feasible
                if feas:

                    # progress message
                    print("z={}\nbeta={}".format(z, self.beta))

                    # store solution
                    zo = z
                    betas = self.beta

                    # record solution
                    sols.append((zo, betas))

                    # finished homotopy
                    if self.beta == betag:
                        print("Finished")
                        break

                    # attempting homotopy boundary
                    elif (step < 0 and self.beta < 0.001) or (step > 0 and self.beta > 0.99):
                        self.beta = betag
                        i += 1

                    # change homotopy parameter
                    else:

                        # bounded step
                        bstep = min(abs(self.beta - (betag + self.beta)/2), abs(step))

                        # apply step 
                        self.beta += bstep if step > 0 else -bstep

                # infeasible
                else:
                    self.beta = (betas + self.beta)/2

            # return solutions
            return np.array(sols)

    def random_walks(self, tl, sll, alpha, Tlb, Tub, lb, dsm=0.02, ns=5, nn=20, nw=10):

        # number of integration nodes
        n = len(tl)

        # indicies
        ind = np.linspace(0, int(n*0.8), ns, dtype=int)

        # sample trajectory
        tls = tl[ind]
        slls = sll[ind]

        # duration for each sample
        Tls = tl[-1] - tls

        # number of cpus
        p = Pool(cpu_count())

        # parallel arguments
        args = [(
            sl[:self.sdim], 
            np.hstack(([T], sl[self.sdim:])),
            alpha,
            nn,
            0, 
            Tub,
            lb
        ) for _ in range(nw) for T, sl in zip(Tls, slls)]

        # map the arguments in parallel
        res = p.starmap(self.random_walk, args)
        return np.concatenate(res)

    def random_walk(self, so, zo, alpha, n, Tlb, Tub, lb, dsm=0.02, verbose=False):
        
        so = np.array(so, float)
        z, f, feas = self.solve(so, alpha, Tlb, Tub, lb=lb, z=zo)
        if feas:
            pass
        else:
            return None

        # trajectory list
        T = list()

        # nominal state perturbation percentage
        ds = float(dsm)

        # random walk
        while len(T) < n:

            # perturb state
            np.random.seed()
            s = np.array(so) + self.sbound*np.random.uniform(-ds, ds, self.sdim)

            # solve
            z, f, feas = self.solve(s, alpha, Tlb, Tub, lb=lb, z=zo)

            # if succesfull
            if feas:

                # set best state and decision
                so = s
                zo = z

                # store solution
                T.append((so, zo))
                
                # increase perturbation size
                ds = (ds + dsm)/2

                # print message
                if verbose:
                    print("Success {}. Step sice now {}".format(len(T), ds))

            # decrease perturbation sice if unsuccesfull
            else:
                ds /= 2

        return T

    def random_walk_par(self, so, zo, alpha, n, Tlb, Tub, lb, nw, dsm=0.2, verbose=False):

        # number of cpus
        p = Pool(cpu_count())

        # parallel arguments
        args = [(so, zo, alpha, n, Tlb, Tub, lb, dsm, verbose) for _ in range(nw)]

        # parallel arguments
        T = p.starmap(self.random_walk, args)

        # return result
        T = np.concatenate(T)
        return np.vstack((np.array((so, zo)), T))
        
    def plot_throttle_homotopy(self, zalpha):

        # create plot
        fig, ax = plt.subplots(1)

        # plot each solution
        for sol in zalpha:

            # decision vector
            z = sol[0]

            # set homotopy parameter
            self.alpha = sol[1]

            # propagate trajectory
            tl, sl, ul = self.propagate(z[0], z[1:], u=True)

            # plot trajectory
            ax.plot(tl, ul, "k-", alpha=0.1)

        return fig, ax

    def plot_states(self, tl, sl, ul):

        # create axis
        fig, ax = plt.subplots(self.sdim + self.udim, sharex=True)

        # plot each state
        for i in range(self.sdim):
            ax[i].plot(tl, sl[:,i], "k-", alpha=0.5)
        ax[-1].plot(tl, ul, "k-", alpha=0.5)
        
        return fig, ax

    def homotopy_db(self, s0zl, alpha, Tlb, Tub, lb, alphag, step=0.01):

        # arguments for parallel - 
        args = [(s0z[0], alpha, Tlb, Tub, lb, s0z[1], alphag, step) for s0z in s0zl]

        # parallel homotopy - z alpha
        zall = Pool(cpu_count()).starmap(self.homotopy, args)

        # assemble results
        res = list()
        for s0z, zal in zip(s0zl, zall):

            # initial state
            s0 = s0z[0]

            # for each homotopy node
            for za in zal:

                # decision vector
                z = za[0]

                # homotopy parameter
                a = za[1]

                # add result
                res.append((s0, z, a))

        return np.array(res)

    def homotopy_db_beta(self, s0zl, alpha, beta, Tlb, Tub, lb, betag, step=0.01):

        # arguments for parallel
        args = [(s0z[0], alpha, beta, Tlb, Tub, lb, s0z[1], betag, step) for s0z in s0zl]

        # parallel homotopy - z alpha
        zall = Pool(cpu_count()).starmap(self.homotopy_beta, args)

        # assemble results
        res = list()
        for s0z, zal in zip(s0zl, zall):

            # initial state
            s0 = s0z[0]

            # for each homotopy node
            for za in zal:

                # decision vector
                z = za[0]

                # homotopy parameter
                a = za[1]

                # add result
                res.append((s0, z, a))

        return np.array(res)

    def gen_db(self, res, cat=False):

        # data list
        dl = list()

        # propagate all the trajectories
        for r in res:
            s0, z, alpha = r
            tll, sll, ul = self.propagate(z[0], s0, z[1:], alpha, u=True)
            d = np.hstack((
                sll[:, :self.sdim],
                np.full((len(sll), 1), alpha), 
                ul.reshape(-1, 1)
            ))
            dl.append(d)
        if cat:
            return np.vstack(dl)
        else:
            return np.array(dl)
