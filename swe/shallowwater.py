#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
from pathos.multiprocessing import ProcessingPool
from functools import partial

from swe.numerics.numerical_flux import LF_flux, rusanov_flux
import swe.numerics.direct_MLT_ADJ_LF as diradj
from swe.animation_SWE import animate_SWE
from swe.numerics.boundary_conditions import bcR, bcL_A, bcR_A, BCrand, BCperiodic
from swe.numerics.interpolation_tools import interp
from swe.cost import cost_observations as cst
import itertools

g = 9.81
eta = 7.0 / 3.0


def BCsin(h, hu, t, mean_h, amplitude_vector, fundperiod, phase):
    """Conditions aux limites du modele direct, avec plus de paramètres"""
    h[0] = mean_h
    period = fundperiod
    for amp in amplitude_vector:
        h[0] += amp * np.sin((t * (2 * np.pi) / period) + phase)
        period /= 2.0
    hu[0] = 0.0
    return [h] + [hu]


class ShallowWaterSimulation:
    def __init__(
        self,
        D=[0, 100],
        T=200,
        dt=0.03,
        N=200,
        h0=lambda x: 20 * np.ones_like(x),
        u0=lambda x: np.zeros_like(x),
        b=None,
        K=None,
        numflux=LF_flux,
        idx_observation=None,
        bcL=None,
        periodic=False,
        external_forcing=None,
    ):
        """Create a ShallowWaterSimulation object

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
        leftBC: list (to be removed ?)
            default parameters to be used in [mean_h, amplitude, period, phase] model
        K: float or array of floats
            bottom friction to be used in the simulation
        numflux: callable
            function that computes the numerical flux in finite volume
        idx_observation: list of int
            where the water height is observed in the cost function evaluation
        bcL: callable
            function defining the boundary conditions at the left of the domain
        periodic: boolean
            periodic boundary conditions to be considered
        external_forcing: callable
            function to be call (as a callback) to modify state vector
        """

        self.D = D
        self.T = T
        self.dt = dt
        self.N = N
        self.h0 = h0
        self.u0 = u0
        self.dx = np.diff(self.D)[0] / float(self.N)
        self.xr = np.linspace(
            self.D[0] + self.dx / 2.0, self.D[1] - self.dx / 2.0, self.N
        )
        self.b = b
        Karray = np.asarray(K)
        K_transform = interp(Karray, D)
        self.Karray = np.array(map(K_transform, self.xr))

        self.bcLeft = bcL
        self.idx_observation = idx_observation
        self.h_reference = None
        self.ssh = None
        self.J_cost = None
        self.periodic = periodic
        self.external_forcing = external_forcing
        self.numflux = numflux

    def summary(self):
        print("\n--------------------------------------------")
        print("     Shallow Water Simulation instance")
        print("--------------------------------------------")
        print("Domain:   D={}, dx={}, Nvolumes={}".format(self.D, self.dx, self.N))
        print(
            "Stop Time T={}, dt={}, Ntimesteps={}".format(
                self.T, self.dt, int(self.T / self.dt) + 1
            )
        )
        if self.periodic is False:
            per = "No"
        else:
            per = "Yes"
        print("Periodic BC: {}".format(per))
        print("Numerical Flux: {}".format(self.numflux.func_name))

    def direct_simulation(self, verbose=False):
        """Performs a direct run of the model specified"""
        if verbose:
            self.summary()
        [xr, h, u, t] = diradj.shallow_water(
            self.D,
            g,
            self.T,
            self.h0,
            self.u0,
            self.N,
            self.numflux,
            self.dt,
            self.b,
            self.Karray,
            self.bcLeft,
            bcR,
            self.periodic,
            external_forcing=self.external_forcing,
            tstart=0.0,
        )
        self.ssh = h
        self.u = u
        return xr, h, u, t

    def continue_simulation(self, tstart, T2, h0=None, u0=None):
        """Continues the direct run of the model, between T and T2"""
        if self.b is not None:
            bathymetry = self.b(self.xr)
        else:
            bathymetry = 0
        if h0 is None:
            h0 = self.ssh[:, -1] + bathymetry
        if u0 is None:
            u0 = self.u[:, -1]
        [xr, h, u, tbis] = diradj.shallow_water(
            self.D,
            g,
            T2,
            h0,
            u0,
            self.N,
            LF_flux,
            self.dt,
            self.b,
            self.Karray,
            self.bcLeft,
            bcR,
            self.periodic,
            external_forcing=self.external_forcing,
            tstart=tstart,
        )
        return xr, h, u, tbis

    def continue_simulation_par(self, Karray, bcL, T2, tstart=None, h0=None, u0=None):
        """Continues the direct run of the model, between T and T2
        with bottom friction and bcL changed
        """
        if self.b is not None:
            bathymetry = self.b(self.xr)
        else:
            bathymetry = 0
        if h0 is None:
            h0 = self.ssh[:, -1] + bathymetry
        if u0 is None:
            u0 = self.u[:, -1]
        if Karray is None:
            Karray = self.Karray
        if tstart is None:
            tstart = self.T
        [xr, h, u, tbis] = diradj.shallow_water(
            self.D,
            g,
            T2,
            h0,
            u0,
            self.N,
            LF_flux,
            self.dt,
            self.b,
            Karray,
            bcL,
            bcR,
            self.periodic,
            external_forcing=self.external_forcing,
            tstart=tstart,
        )
        return xr, h, u, tbis

    def set_reference(self, h_reference):
        self.h_reference = h_reference
        if self.idx_observation is not None:
            self.cost_fun, self.obs_mat = cst.J_function_observation_init(
                self.h_reference, self.idx_observation
            )

    def compute_cost_and_gradient(self):
        """
        Compute cost and gradient between reference observation, and instance
        """
        return diradj.shallow_water_RSS_grad_observation(
            self.D,
            g,
            self.T,
            self.h0,
            self.u0,
            self.N,
            LF_flux,
            self.dt,
            self.b,
            self.Karray,
            self.bcLeft,
            bcR,
            bcL_A,
            bcR_A,
            self.h_reference,
            self.cost_fun,
            self.obs_mat,
        )

    def compute_cost(self):
        """
        Compute Cost between reference observations, and current instance
        """
        self.direct_simulation()
        cost, _ = cst.J_function_observation(
            self.ssh, self.h_reference, self.idx_observation
        )
        return cost


class CostSWE:
    def __init__(self, ref, bcL_U):
        """ """
        self.ref = ref
        _, self.obs_mat = cst.J_function_observation_init(
            self.ref.ssh, self.ref.idx_observation
        )
        self.bcL = bcL_U

    def create_simulation(self, K, U):
        sr = self.ref

        def bc_for_sim(h, hu, t):
            return self.bcL(h, hu, t, U)

        swesim = ShallowWaterSimulation(
            D=sr.D,
            T=sr.T,
            dt=sr.dt,
            N=sr.N,
            h0=sr.h0,
            u0=sr.u0,
            b=sr.b,
            K=K,
            idx_observation=sr.idx_observation,
            bcL=bc_for_sim,
            periodic=sr.periodic,
            external_forcing=sr.external_forcing,
            numflux=sr.numflux,
        )
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
        sizeK = instance_to_compare.Karray.size
        grad_sum_length = int(instance_to_compare.xr.shape[0] / sizeK)
        grad_coeff = np.zeros(sizeK)
        for i in range(sizeK):
            grad_coeff[i] = sum(
                grad[i * grad_sum_length : (i * grad_sum_length + grad_sum_length)]
            )
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
                print("Not enough points to compute, switch to unparallelized proc")
                trigger = True

        start_fun = time.time()
        if trigger:
            response = np.empty(npoints)
            if adj_gradient and isinstance(KU[0, 0], float):
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
                        np.asarray(arg_K), points[1:]
                    )
                else:
                    response[i] = fun_to_evaluate(np.asarray(arg_K), points[1:])
                if verbose:
                    print(
                        "Time elapsed for unparallelized computations: {}".format(
                            time.time() - start
                        )
                    )
        else:  # Parallelized computations
            try:
                pool = ProcessingPool(nodes=ncores)
            except AssertionError:
                pool.restart()
            split = np.array_split(KU, ncores, 0)
            start = time.time()
            if adj_gradient:
                try:
                    response_i, gradient_i = zip(
                        *pool.map(
                            partial(
                                self.J_KU,
                                parallel=False,
                                adj_gradient=adj_gradient,
                                verbose=False,
                            ),
                            split,
                        )
                    )
                except AssertionError:
                    pool.restart()
                    response_i, gradient_i = zip(
                        *pool.map(
                            partial(
                                self.J_KU,
                                parallel=False,
                                adj_gradient=adj_gradient,
                                verbose=False,
                            ),
                            split,
                        )
                    )
            else:
                try:
                    response_i = pool.map(
                        partial(
                            self.J_KU,
                            parallel=False,
                            adj_gradient=adj_gradient,
                            verbose=False,
                        ),
                        split,
                    )
                except AssertionError:
                    pool.restart()
                    response_i = pool.map(
                        partial(
                            self.J_KU,
                            parallel=False,
                            adj_gradient=adj_gradient,
                            verbose=False,
                        ),
                        split,
                    )

                    # array_split necessary ?
            print(
                "Time elapsed for parallelized computations: {}".format(
                    time.time() - start
                )
            )
            response = np.asarray([item for sublist in response_i for item in sublist])
            if adj_gradient:
                gradient = np.asarray(
                    [item for sublist in gradient_i for item in sublist]
                )
            pool.close()
            pool.join()

        if verbose:
            print(
                "Mean time for computation: {}".format(
                    (time.time() - start_fun) / npoints
                )
            )
        if adj_gradient:
            return response, gradient
        else:
            return response

    def comparison_predictions(
        self, Karray, u, utrue, Tpred, BCtrue, BCsim=None, parallel=True, ncores=None
    ):
        if BCsim is None:
            BCsim = self.bcL
        if ncores is None:
            ncores = cpu_count()
        pred = []
        for u_t in utrue:
            if parallel:
                try:
                    pool = ProcessingPool(nodes=ncores)
                except AssertionError:
                    pool.restart()
            print("u start {}".format(u_t))

            def bcLtrue(h, q, t):
                return BCtrue(h, q, t, u_t)

            _, htruepred, _, _ = self.ref.continue_simulation_par(
                Karray=None, bcL=bcLtrue, T2=Tpred
            )

            def continuation(ku_s, single=False):
                print("Continuation start")
                if single:
                    k, u_sim = ku_s
                    Ka = np.array(map(interp(k, self.ref.D), self.ref.xr))

                    def bcL(h, q, t):
                        return BCsim(h, q, t, u_sim)

                    _, hpred, _, _ = self.ref.continue_simulation_par(
                        Karray=np.asarray(Ka), bcL=bcL, T2=Tpred
                    )
                    cost = np.sum((htruepred - hpred) ** 2)
                else:
                    cost = np.empty(len(ku_s))
                    for i, ku_iter in enumerate(ku_s):
                        k, u_sim = ku_iter
                        Ka = np.array(map(interp(k, self.ref.D), self.ref.xr))

                        def bcL(h, q, t):
                            return BCsim(h, q, t, u_sim)

                        _, hpred, _, _ = self.ref.continue_simulation_par(
                            Karray=np.asarray(Ka), bcL=bcL, T2=Tpred
                        )
                        cost[i] = np.sum((htruepred - hpred) ** 2)
                return cost

            points = list(itertools.product(*[Karray, u]))
            if not parallel:
                for k, u_sim in points:
                    cost = continuation((k, u_sim), single=True)
                    pred.append([u_t, u_sim, k, cost])

            elif parallel:
                print("Parallelization start")

                split = np.array_split(points, ncores, 0)
                try:
                    pool = ProcessingPool(nodes=ncores)
                    pred_i = pool.map(continuation, split)

                except AssertionError:
                    pool.restart()
                    pred_i = pool.map(continuation, split)

                # pool.close()
                pred_i = np.asarray([item for sublist in pred_i for item in sublist])
                pred.append(pred_i)
        if parallel:
            pool.close()
            pool.join()
            properties = np.asarray(
                [[ut, u_, K] for ut in utrue for K in Karray for u_ in u]
            )
            pred = np.asarray(pred).flatten()
            pred = np.hstack([properties, np.atleast_2d(pred).T])
        return np.asarray(pred)


def main():
    """
    Main script to execute for testing purpose or prototyping
    """
    # bcsin_ref = lambda h, hu, t: BCsin(h, hu, t, 16, [1, 5, 0.5, 0.25], fundperiod=20,
    #                                    phase=0)

    # # bcsumsin_sim = lambda h, hu, t: BCsumsin(h, hu, t, 'L', 16, [5, 0.7, 0, 0.3], 5, 0)

    # D = [0, 100]
    # N = 200  # Nombre de volumes
    # dx = np.diff(D)[0] / float(N)  # Largeur des volumes
    # xr = np.linspace(D[0] + dx / 2, D[1] - dx / 2, N)  # Milieux des volumes
    # b = lambda x: 10 * x / D[1] + 0.5 * np.cos(x / 3) + 3 / (1 + np.exp(-3 * (x - 50)))
    # T = 50
    # dt = 0.03
    # Kref = 0.2 * (1 + np.sin(2 * np.pi * xr / D[1]))

    # ref = ShallowWaterSimulation(T=T, b=b, K=Kref, dt=dt,
    #                              idx_observation=np.arange(49, 200, 50, dtype=int),
    #                              bcL=bcsin_ref)
    # ref.direct_simulation()

    # def bc_example(h, hu, t, U):
    #     amplitude_vector = [1, U, 0.5, 0.25]
    #     return BCsin(h, hu, t, 16, amplitude_vector, fundperiod=20,
    #                  phase=0)

    # model = CostSWE(ref, bc_example)

    # KU = np.atleast_2d([[0.1, 0.5],
    #                     [0.1, 1.0],
    #                     [0.1, 1.5],
    #                     [0.2, 0.5],
    #                     [0.2, 1.0],
    #                     [0.2, 1.5],
    #                     [0.0, 0.5],
    #                     [0.0, 1.0],
    #                     [0.0, 2.0],
    #                     [0.05, 0.0],
    #                     [0.05, 1.0],
    #                     [0.2, 1.5],
    #                     [0.0, 0.5],
    #                     [0.0, 1.0],
    #                     [0.0, 2.0],
    #                     [0.05, 0.0],
    #                     [0.05, 1.0]])
    # model.J_KU(KU, adj_gradient=False, parallel=True)

    # Periodic BC
    D = [0, 500]
    N = 500  # Nombre de volumes
    dx = np.diff(D)[0] / float(N)  # Largeur des volumes
    xr = np.linspace(D[0] + dx / 2, D[1] - dx / 2, N)  # Milieux des volumes

    def b(x):
        return np.zeros_like(x)

    T = 100
    dt = 0.001
    Kref = 0.2 * (1 + np.sin(2 * np.pi * xr / D[1]))

    def external_forcing(h, hu, t):
        """
        Must return h, hu
        """
        # c = 300
        # add = np.exp(-(np.mod(xr - c * t, D[1]))**2 / 20)
        # add = np.sin(xr * np.pi / (D[1])) * np.sin(t / 2.0)
        # add2 = np.sin(2 * xr * np.pi / (D[1])) * np.sin(t / 1.0)
        hu[0] = (
            hu[0]
            + np.sin(t * np.pi / 1.0)
            + 0.3 * np.sin(3 * t * np.pi)
            + 0.1 * np.sin(t * np.pi / 6.0)
        )
        # h = h + 0.5 * add# + 0.01 * add2
        return h, hu

    def external_forcing2(h, hu, t):
        """
        Must return h, hu
        """
        hu[0] = (
            hu[0] + np.sin(t * np.pi / 1.0) + 0.3 * np.sin(3 * t * np.pi)
        )  # + 0.1 * np.sin(t * np.pi / 6.0)
        return h, hu

    def bathy(x):
        return 1.0 * (1 - np.cos(2 * np.pi * x / D[1])) / 2.0

    def h0(x):
        return 5 * np.ones_like(x) + 10 * np.exp(-((x - 50) ** 2) / 50)

    ref = ShallowWaterSimulation(
        T=T,
        b=bathy,
        K=Kref,
        dt=dt,
        h0=h0(xr),
        N=N,
        bcL=BCperiodic,
        periodic=True,
        external_forcing=external_forcing,
        numflux=rusanov_flux,
    )
    ref2 = ShallowWaterSimulation(
        T=T,
        b=bathy,
        K=Kref,
        dt=dt,
        h0=h0(xr),
        N=N,
        bcL=BCperiodic,
        periodic=True,
        external_forcing=external_forcing2,
        numflux=rusanov_flux,
    )

    _ = ref.direct_simulation()
    _ = ref2.direct_simulation()

    animate_SWE(xr, [ref.ssh, ref2.ssh], bathy, D, ylim=[0, 30])
    animate_SWE(
        np.linspace(0, 1000, 2000),
        [np.vstack([ref.ssh, ref.ssh])],
        b=bathy,
        D=[0, 1000],
        ylim=[0, 30],
    )


if __name__ == "__main__":
    main()
