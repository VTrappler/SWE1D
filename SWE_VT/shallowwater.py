#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import time
from multiprocessing import Pool, cpu_count
from pathos.multiprocessing import ProcessingPool
from functools import partial

from SWE_VT.numerics.LaxFriedrichs import LF_flux
import SWE_VT.numerics.direct_MLT_ADJ_LF as diradj
from SWE_VT.animation_SWE import animate_SWE
from SWE_VT.numerics.boundary_conditions import (bcR,
                                                 bcL_A, bcR_A,
                                                 BCrand)
from SWE_VT.numerics.interpolation_tools import interp
from SWE_VT.cost import cost_observations as cst

g = 9.81
eta = 7. / 3.


def BCsin(h, hu, t, mean_h, amplitude_vector, fundperiod, phase):
    """ Conditions aux limites du modele direct, avec plus de paramètres"""
    h[0] = mean_h
    period = fundperiod
    for amp in amplitude_vector:
        h[0] += amp * np.sin((t * (2 * np.pi) / period) + phase)
        period /= 2.0
    hu[0] = 0.0
    return [h] + [hu]


class ShallowWaterSimulation:
    def __init__(self,
                 D=[0, 100],
                 T=200,
                 dt=0.03,
                 N=200,
                 h0=lambda x: 20 * np.ones_like(x),
                 u0=lambda x: np.zeros_like(x),
                 b=None,
                 leftBC=[20, 0, 5.0, 15.0],
                 K=None,
                 idx_observation=None,
                 bcL=None):
        """ Create a ShallowWaterSimulation object

        The shallow water model is initialized with the following parameters:

        Parameters
        ----------
        D: list
            x coordinate of the left and right boundary
        T: float
            Time of the end of the simulation
        dt: float
            Timestep
        N: int
            Number of volumes
        h0: callable
            function of the initial water height
        u0: callable
            function of the initial water discharge
        b: callable
            function defining the bathymetry
        leftBC: list
            default parameters to be used in [mean_h, amplitude, period, phase] model
        K: float or array of floats
            bottom friction to be used in the simulation
        idx_observation: list of int
            where the water height is observed in the cost function evaluation
        bcL: callable
            function defining the boundary conditions at the left of the domain
        """

        self.D = D
        self.T = T
        self.dt = dt
        self.N = N
        self.h0 = h0
        self.u0 = u0
        self.dx = np.diff(self.D)[0] / float(self.N)
        self.xr = np.linspace(self.D[0] + self.dx / 2.0, self.D[1] - self.dx / 2.0, self.N)
        self.b = b
        Karray = np.asarray(K)
        K_transform = interp(Karray, D)
        self.Karray = np.array(map(K_transform, self.xr))
        self.Kcoeff = np.asarray(K)
        if bcL is None:
            self.bcLeft = lambda h, hu, t: BCrand(h, hu, t, 'L',
                                                  leftBC[0], leftBC[1], leftBC[2], leftBC[3])
        else:
            self.bcLeft = bcL
        self.idx_observation = idx_observation
        self.h_reference = None
        self.ssh = None
        self.J_cost = None


    def direct_simulation(self):
        """ Performs a direct run of the model specified

        """
        [xr, h, u, t] = diradj.shallow_water(self.D, g, self.T,
                                             self.h0, self.u0,
                                             self.N, LF_flux,
                                             self.dt, self.b,
                                             self.Karray, self.bcLeft,
                                             bcR)
        self.ssh = h
        self.u = u
        return xr, h, u, t


    def continue_simulation(self, T2, h0=None, u0=None):
        """ Continues the direct run of the model, between T and T2
        """
        if self.b is not None:
            to_add = self.b(self.xr)
        else:
            to_add = 0
        if h0 is None:
            h0 = self.ssh[:, -1] + to_add
        if u0 is None:
            u0 = self.u[:, -1]
        [xr, h, u, tbis] = diradj.shallow_water(self.D, g, T2,
                                                h0, u0,
                                                self.N, LF_flux,
                                                self.dt, self.b,
                                                self.Karray, self.bcLeft,
                                                bcR)
        return xr, h, u, tbis


    def set_reference(self, h_reference):
        self.h_reference = h_reference
        if self.idx_observation is not None:
            self.cost_fun, self.obs_mat = cst.J_function_observation_init(self.h_reference,
                                                                          self.idx_observation)


    def compute_cost_and_gradient(self):
        """
        Compute cost and gradient between reference observation, and instance
        """
        return diradj.shallow_water_RSS_grad_observation(self.D, g,
                                                         self.T,
                                                         self.h0, self.u0,
                                                         self.N, LF_flux,
                                                         self.dt, self.b, self.Karray,
                                                         self.bcLeft, bcR, bcL_A, bcR_A,
                                                         self.h_reference,
                                                         self.cost_fun,
                                                         self.obs_mat)

    def compute_cost(self):
        """
        Compute Cost between reference observations, and current instance
        """
        self.direct_simulation()
        cost, _ = cst.J_function_observation(self.ssh, self.h_reference, self.idx_observation)
        return cost



class CostSWE:
    def __init__(self, ref):
        """

        """
        self.ref = ref
        _, self.obs_mat = cst.J_function_observation_init(self.ref.ssh,
                                                          self.ref.idx_observation)


    def create_simulation(self, K, U):
        sr = self.ref
        amplitude_vector = [0, U, 0.5, 0.25]
        bcsinsim = lambda h, hu, t: BCsin(h, hu, t, 16, amplitude_vector, fundperiod=20,
                                          phase=0)
        swesim = ShallowWaterSimulation(D=sr.D,
                                        T=sr.T,
                                        dt=sr.dt,
                                        N=sr.N,
                                        h0=sr.h0,
                                        u0=sr.u0,
                                        b=sr.b,
                                        leftBC=None,
                                        K=K,
                                        idx_observation=sr.idx_observation,
                                        bcL=bcsinsim)
        swesim.set_reference(self.ref.ssh)
        return swesim


    def J(self, K, U):
        K = np.asarray(K)
        instance_to_compare = self.create_simulation(K, U)
        return instance_to_compare.compute_cost()


    def JG(self, K, U):
        K = np.asarray(K)
        instance_to_compare = self.create_simulation(K, U)
        # instance_to_compare.direct_simulation()
        cost, grad = instance_to_compare.compute_cost_and_gradient()
        sizeK = instance_to_compare.Kcoeff.size
        grad_sum_length = int(instance_to_compare.xr.shape[0] / sizeK)
        grad_coeff = np.zeros(sizeK)
        for i in range(sizeK):
            grad_coeff[i] = sum(grad[i * grad_sum_length:(i * grad_sum_length + grad_sum_length)])
        return cost, grad_coeff


    def J_KU(self, KU, parallel=False, ncores=None, adj_gradient=False, verbose=True):
        if adj_gradient:
            fun_to_evaluate = self.JG
        else:
            fun_to_evaluate = self.J

        npoints = KU.shape[0]
        trigger = True  # True for unparallelized
        if parallel:
            trigger = False
            if ncores is None:
                ncores = cpu_count()
            if npoints < 10:
                print('Not enough points to compute, switch to unparallelized proc')
                trigger = True

        start_fun = time.time()
        if trigger:
            response = np.empty(npoints)
            if (adj_gradient and isinstance(KU[0, 0], float)):
                gradient = np.empty([npoints, 1])
            elif adj_gradient:
                gradient = np.empty([npoints, len(KU[0, 0])])

            for i, points in enumerate(KU):
                start = time.time()
                if isinstance(points[0], float):
                    arg_K = [points[0]]
                else:
                    arg_K = points[0]

                if adj_gradient:
                    response[i], gradient[i, :] = fun_to_evaluate(
                        np.asarray(arg_K), points[1:])
                else:
                    response[i] = fun_to_evaluate(
                        np.asarray(arg_K), points[1:])
                if verbose:
                    print('Time elapsed for unparallelized computations: {}'.
                          format(time.time() - start))



        else:  # Parallelized computations
            try:
                pool = ProcessingPool(nodes = ncores)
            except AssertionError:
                pool.restart()
            split = np.array_split(KU, ncores, 0)
            start = time.time()
            if adj_gradient:
                try:
                    response_i, gradient_i = zip(*pool.map(partial(self.J_KU,
                                                                   parallel=False,
                                                                   adj_gradient=adj_gradient,
                                                                   verbose=False),
                                                           split))
                except AssertionError:
                    pool.restart()
                    response_i, gradient_i = zip(*pool.map(partial(self.J_KU,
                                                                   parallel=False,
                                                                   adj_gradient=adj_gradient,
                                                                   verbose=False),
                                                           split))
            else:
                try:
                    response_i = pool.map(partial(self.J_KU,
                                                  parallel=False,
                                                  adj_gradient=adj_gradient,
                                                  verbose=False),
                                          split)
                except AssertionError:
                    pool.restart()
                    response_i = pool.map(partial(self.J_KU,
                                                  parallel=False,
                                                  adj_gradient=adj_gradient,
                                                  verbose=False),
                                          split)

                    # array_split necessary ?
            print('Time elapsed for parallelized computations: {}'.
                  format(time.time() - start))
            response = np.asarray([item for sublist in response_i
                                   for item in sublist])
            if adj_gradient:
                gradient = np.asarray([item for sublist in gradient_i
                                       for item in sublist])
            pool.close()
            pool.join()


        if verbose:
            print('Mean time for computation: {}'.format((time.time() - start_fun) / npoints))
        if adj_gradient:
            return response, gradient
        else:
            return response


def main():

    bcsin_ref = lambda h, hu, t: BCsin(h, hu, t, 16, [1, 5, 0.5, 0.25], fundperiod=20,
                                       phase=0)

    # bcsumsin_sim = lambda h, hu, t: BCsumsin(h, hu, t, 'L', 16, [5, 0.7, 0, 0.3], 5, 0)

    D = [0, 100]
    N = 200  # Nombre de volumes
    dx = np.diff(D)[0] / float(N)  # Largeur des volumes
    xr = np.linspace(D[0] + dx / 2, D[1] - dx / 2, N)  # Milieux des volumes
    b = lambda x: 10 * x / D[1] + 0.5 * np.cos(x / 3) + 3 / (1 + np.exp(-3 * (x - 50)))
    T = 50
    dt = 0.03
    Kref = 0.2 * (1 + np.sin(2 * np.pi * xr / D[1]))

    ref = ShallowWaterSimulation(T=T, b=b, K=Kref, dt=dt,
                                 idx_observation=np.arange(49, 200, 50, dtype=int),
                                 bcL=bcsin_ref)
    ref.direct_simulation()

    model = CostSWE(ref)
    KU = np.atleast_2d([[0.1, 0.5],
                        [0.1, 1.0],
                        [0.1, 1.5],
                        [0.2, 0.5],
                        [0.2, 1.0],
                        [0.2, 1.5],
                        [0.0, 0.5],
                        [0.0, 1.0],
                        [0.0, 2.0],
                        [0.05, 0.0],
                        [0.05, 1.0],
                        [0.2, 1.5],
                        [0.0, 0.5],
                        [0.0, 1.0],
                        [0.0, 2.0],
                        [0.05, 0.0],
                        [0.05, 1.0]])
    model.J_KU(KU, adj_gradient=False, parallel=True)






    test = ShallowWaterSimulation(T=T, b=b, K=[0.1],
                                  idx_observation=np.arange(49, 200, 50, dtype=int),
                                  bcL=bcsin_sim)
    test.direct_simulation()

    animate_SWE(xr, [href, test.ssh], b, D, [0, 50])

    _, h50, u50, tbis = test.continue_simulation(50)

    test.set_reference(href)
    J, G = test.compute_cost()

    def J_K(K):
        K = np.asarray([K])
        test = ShallowWaterSimulation(T=T, b=b, K=K,
                                      idx_observation=np.arange(49, 200, 50, dtype=int),
                                      bcL=bcsin_sim)
        test.set_reference(href)
        J, G = test.compute_cost()
        return J
    rss = []
    for k in np.linspace(0, 0.25, 15):
        rss.append(J_K(k))
    import matplotlib.pyplot as plt
    plt.plot(rss)
    plt.show()
    test = CostSWE(ref)
    start = time.time()
    test.J([0.2], 5)
    interm = time.time()
    print(interm - start)
    test.JG([0.2], 5)
    print(time.time() - interm)


if __name__ == '__main__':
    main()
