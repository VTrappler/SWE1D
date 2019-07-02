#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import time
from multiprocessing import Pool, cpu_count
from functools import partial

from code_swe.LaxFriedrichs import LF_flux
import code_swe.direct_MLT_ADJ_LF as diradj
import HR_config.wrapper as cost
from code_swe.animation_SWE import animate_SWE
from code_swe.boundary_conditions import bcL, bcR, bcL_d, bcR_d, bcL_A, bcR_A, BCrand, BCsumsin
from code_swe.interpolation_tools import interp_cst


g = 9.81
eta = 7. / 3.



class SWEsim:
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
        self.D = D
        self.T = T
        self.dt = dt
        self.N = N
        self.h0 = h0
        self.u0 = u0
        self.dx = np.diff(self.D)[0] / float(self.N)
        self.xr = np.linspace(self.D[0] + self.dx / 2.0, self.D[1] - self.dx / 2.0, self.N)
        self.b = b
        self.K = K
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
        [xr, h, u, t] = diradj.shallow_water(self.D, g, self.T,
                                             self.h0, self.u0,
                                             self.N, LF_flux,
                                             self.dt, self.b,
                                             self.K, self.bcLeft,
                                             bcR)
        self.ssh = h
        self.u = u
        return xr, h, u, t


    def continue_simulation(self, T2, h0=None, u0=None):
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
                                                self.K, self.bcLeft,
                                                bcR)
        return xr, h, u, tbis


    def set_reference(self, h_reference):
        self.h_reference = h_reference
        if self.idx_observation is not None:
            self.cost_fun, self.obs_mat = cost.J_function_observation_init(self.h_reference,
                                                                           self.idx_observation)


    def compute_cost(self):
        """
        Compute cost and gradient between reference observation, and instance
        """
        return diradj.shallow_water_RSS_grad_observation(self.D, g,
                                                         self.T,
                                                         self.h0, self.u0,
                                                         self.N, LF_flux,
                                                         self.dt, self.b, self.K,
                                                         self.bcLeft, bcR, bcL_A, bcR_A,
                                                         self.h_reference,
                                                         self.cost_fun,
                                                         self.obs_mat)

    def J_KAP(self, Kcoeff, A, P):
        return cost.J_KAP(Kcoeff, A, P, idx_to_observe=self.idx_observation, hreference=self.ssh)



def main():

    bcsumsin_ref = lambda h, hu, t: BCsumsin(h, hu, t, 'L', 16, [5, 1, 0.5, 0.25], 5, 0)
    bcsumsin_sim = lambda h, hu, t: BCsumsin(h, hu, t, 'L', 16, [5, 0.7,  0, 0.3 ], 5, 0)

    D = [0, 100]
    N = 200  # Nombre de volumes
    dx = np.diff(D)[0] / float(N)  # Largeur des volumes
    xr = np.linspace(D[0] + dx / 2, D[1] - dx / 2, N)  # Milieux des volumes
    b = lambda x: 10 * x / D[1] + 0.5 * np.cos(x / 3) + 3 / (1 + np.exp(-3 * (x - 50)))
    T = 50
    Kref = 0.2 * (1 + np.sin(2 * np.pi * xr / D[1]))
    MAPPh = [15.0, 5.0, 5.0, 0]

    ref = SWEsim(T=T, b=b, K=Kref, leftBC=MAPPh,
                 idx_observation=np.arange(49, 200, 50, dtype=int), bcL = bcsumsin_ref)
    href = ref.direct_simulation()[1]
    test = SWEsim(T=T, leftBC=MAPPh, b=b, K=0.1,
                  idx_observation=np.arange(49, 200, 50, dtype=int),
                  bcL=bcsumsin_sim)
    test.direct_simulation()
    _, h50, u50, tbis = test.continue_simulation(50)
    animate_SWE(xr, [href, test.ssh], b, D, [0, 50])
    test.set_reference(href)
    J, G = test.compute_cost()

    def J_K(K):
        test = SWEsim(T=T, leftBC=MAPPh, b=b, K=K,
                      idx_observation=np.arange(49, 200, 50, dtype=int),
                      bcL=bcsumsin_sim)
        test.set_reference(href)
        J, G = test.compute_cost()
        return J
    J_vec = np.empty(20)
    for i, k in enumerate(np.linspace(0, 0.5, 20)):
        J_vec[i] = J_K(k)

    
    # test.J_KAP([1.0, 0.5, 1.0], 4.9, 15.1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.plot(J_vec)
    plt.show()
    main()
