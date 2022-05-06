#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------
#           WRAPPER for high-resolution toy problem of SWE
# ------------------------------------------------------------------------------

import numpy as np
import time
from multiprocessing import Pool, cpu_count
from functools import partial

from swe.numerics.LaxFriedrichs import LF_flux
import swe.numerics.direct_MLT_ADJ_LF as diradj
from swe.animation_SWE import animate_SWE
from swe.numerics.boundary_conditions import (
    bcL,
    bcR,
    bcL_d,
    bcR_d,
    bcL_A,
    bcR_A,
    BCrand,
)
from swe.numerics.interpolation_tools import interp_cst


# -- Paramètres du modèle -----------------------------------------------------

D = [0, 100]  # Domaine spatial
T = 60  # Fin de la simulation
dt = 0.03  # Pas de temps
g = 9.81  # Constante de gravitation
N = 200  # Nombre de volumes
dx = np.diff(D)[0] / float(N)  # Largeur des volumes
xr = np.linspace(D[0] + dx / 2, D[1] - dx / 2, N)  # Milieux des volumes
eta = 7.0 / 3.0

# Conditions initiales
u0 = lambda x: 0 * x
h0 = lambda x: np.where(x > np.mean(D), 20, 20)

# Bathymétrie
b = lambda x: 10 * x / D[1] + 0.5 * np.cos(x / 3) + 3 / (1 + np.exp(-3 * (x - 50)))

# Paramétrisation des conditions aux bords à gauche du domaine
mean_h = 20
phase = 0
amplitude = 5.0
period = 15.0

# Valeur de référence de la friction de fond
Kref = 0.2 * (1 + np.sin(2 * np.pi * xr / D[1]))

# Indices à observer
idx_to_observe = np.arange(49, 200, 50, dtype=int)

#
#


# -- Définition des fonctions coûts -------------------------------------------
def J_function(h1, h2):  # Fonction coût
    return 0.5 * np.sum((h1 - h2) ** 2)


# ------------------------------------------------------------------------------
def J_function_observation(h, href, ind_to_observe=None):
    """Compute J, with the index of grid points observed"""
    space_dim = int(h.shape[0])
    if ind_to_observe is None:
        ind_to_observe = range(space_dim)
    obs_mat = np.zeros((href.shape[0], href.shape[0]))
    obs_mat[ind_to_observe, ind_to_observe] = 1
    J = 0.5 * np.sum(obs_mat.dot(h - href) ** 2)
    return J, obs_mat


# ------------------------------------------------------------------------------
def observation_matrix_init(h_test, ind_to_observe=None):
    if ind_to_observe is None:
        ind_to_observe = range(int(h_test.shape[0]))
    obs_mat = np.zeros((h_test.shape[0], h_test.shape[0]))
    obs_mat[ind_to_observe, ind_to_observe] = 1
    return obs_mat


# ------------------------------------------------------------------------------
def J_function_observation_init(h_test, ind_to_observe=None):
    obs_mat = observation_matrix_init(h_test, ind_to_observe)
    J = lambda h1, h2: 0.5 * np.sum(obs_mat.dot(h1 - h2) ** 2)
    return J, obs_mat


# def J_function_regul(h1, h2, alpha, K1, Kb):
#     return 0.5*np.sum((h1-h2)**2) + 0.5*alpha*(K1-Kb)**2

# Ap_Pm -> A = 5.1; P = 49.8
# Ap_Pp -> A = 5.2; P = 50.1
# Am_Pp -> A = 4.9; P = 50.2
# Am_Pm -> A = 4.8; P = 49.9

print("Computation of href performed w/ following parameters:")
print("- Hauteur eau moyenne = ", mean_h)
print("- Amplitude = ", amplitude)
print("- Periode = ", period)
print("- Phase = ", phase)
bcLref = lambda h, hu, t: BCrand(h, hu, t, "L", mean_h, amplitude, period, phase)


# -- Définition des fonctions de simulations --------------------------------------

reference = lambda K: diradj.shallow_water(
    D, g, T, h0, u0, N, LF_flux, dt, b, K, bcLref, bcR
)
direct = lambda K: diradj.shallow_water(
    D, g, T, h0, u0, N, LF_flux, dt, b, K, bcLref, bcR
)
MLT = lambda h, u, K, dK, href: diradj.lineaire_tangent_shallow_water(
    D, g, T, N, dt, b, K, dK, h, u, href, bcL_d, bcR_d
)
ADJ = lambda h, u, K, href: diradj.adjoint_shallow_water(
    D, g, T, N, dt, b, K, h, u, h - href, bcL_A, bcR_A
)


def swe_KAP(K, A, P):
    BC_AP = lambda h, hu, t: BCrand(h, hu, t, "L", mean_h, A, P, phase)
    return diradj.shallow_water(D, g, T, h0, u0, N, LF_flux, dt, b, K, BC_AP, bcR)


# -- Simulation de référence -----------------------------------------------------
[xr, href, uref, t] = reference(Kref)


Nobs = np.prod(href.shape)
# idx_to_observe = range(25)
# cost_fun,obs_mat = J_function_observation_init(href, idx_to_observe)


# ------------------------------------------------------------------------------
def J_grad(K, hreference=href):
    """Effectue une simulation directe, puis utilise la méthode adjointe pour calculer le gradient
    Renvoie la valeur de la fonction coût, ainsi que le gradient en K
    """
    return diradj.shallow_water_RSS_grad(
        D,
        g,
        T,
        h0,
        u0,
        N,
        LF_flux,
        dt,
        b,
        K,
        bcLref,
        bcR,
        bcL_A,
        bcR_A,
        hreference,
        J_function,
    )


# ------------------------------------------------------------------------------
def J_grad_AP(K, ampli, period, idx_to_observe=np.arange(href.shape[0])):
    bcL = lambda h, hu, t: BCrand(h, hu, t, "L", mean_h, ampli, period, phase)
    obs_mat = observation_matrix_init(href, ind_to_observe=idx_to_observe)

    return diradj.shallow_water_RSS_grad_observation(
        D,
        g,
        T,
        h0,
        u0,
        N,
        LF_flux,
        dt,
        b,
        K,
        bcL,
        bcR,
        bcL_A,
        bcR_A,
        href,
        J_function,
        obs_mat,
    )


# ------------------------------------------------------------------------------
def J_grad_AP_href(K, ampli, period, hr):
    bcL = lambda h, hu, t: BCrand(h, hu, t, "L", mean_h, ampli, period, phase)
    return diradj.shallow_water_RSS_grad(
        D, g, T, h0, u0, N, LF_flux, dt, b, K, bcL, bcR, bcL_A, bcR_A, hr, J_function
    )


# ------------------------------------------------------------------------------
def J_naive(K):
    """Calcul RSS (sans gradient) entre href et simulation avec paramètre K"""
    return diradj.shallow_water_RSS(
        D, g, T, h0, u0, N, LF_flux, dt, b, K, bcL, bcR, href, J_function
    )


# ------------------------------------------------------------------------------
def J_obs_grad_base(K, cost_fun, obs_mat):
    """Calcul RSS avec gradient, prenant en arg les observations faites"""
    return diradj.shallow_water_RSS_grad_observation(
        D,
        g,
        T,
        h0,
        u0,
        N,
        LF_flux,
        dt,
        b,
        K,
        bcL,
        bcR,
        bcL_A,
        bcR_A,
        href,
        cost_fun,
        obs_mat,
    )


# ------------------------------------------------------------------------------
def J_grad_observed(
    K, A=amplitude, P=period, idx_to_observe=np.arange(href.shape[0]), hreference=href
):
    """Calcul RSS selon index observés"""
    cost_fun, obs_mat = J_function_observation_init(hreference, idx_to_observe)

    bcL = lambda h, hu, t: BCrand(h, hu, t, "L", mean_h, A, P, phase)
    return diradj.shallow_water_RSS_grad_observation(
        D,
        g,
        T,
        h0,
        u0,
        N,
        LF_flux,
        dt,
        b,
        K,
        bcL,
        bcR,
        bcL_A,
        bcR_A,
        hreference,
        cost_fun,
        obs_mat,
    )


# ------------------------------------------------------------------------------
def J_grad_scalar(listofK):
    """K scalaire pour effectuer calculs en batch"""
    Jvec = 0.0 * np.ones_like(listofK)
    gradvec = 0.0 * np.ones_like(listofK)
    for i, K in enumerate(listofK):
        Jvec[i], gradvec[i] = J_grad(K)
    return [Jvec] + [gradvec]


# ------------------------------------------------------------------------------
@np.vectorize
def likelihood(K, ampli=amplitude, period=period):
    # bcL = lambda h, hu, t: BCrand(h, hu, t, 'L',
    #                              mean_h, ampli, period, phase)
    negloglik = (
        diradj.shallow_water_RSS(
            D,
            g,
            T,
            h0,
            u0,
            N,
            LF_flux,
            dt,
            b,
            K,
            lambda h, hu, t: BCrand(h, hu, t, "L", mean_h, ampli, period, phase),
            bcR,
            href,
            J_function,
        )
        / Nobs
    )
    return np.exp(-negloglik)


# ------------------------------------------------------------------------------
def interp(coef_array):
    """Interpolation using piecewise constant values"""
    coef_array = np.array(coef_array)
    D_length = float(np.diff(D)[0])
    cell_vol = D_length / coef_array.size
    pts = np.linspace(cell_vol / 2.0, D_length - cell_vol / 2.0, num=coef_array.size)
    f_to_ret = lambda x: interp_cst(x, cell_vol, coef_array, pts)
    return f_to_ret


# ------------------------------------------------------------------------------
def J_KAP(
    Kcoeff=Kref,
    A=amplitude,
    P=period,
    idx_to_observe=np.arange(href.shape[0]),
    hreference=href,
):
    """
    Computes RSS and its gradient wrt to K, with amplitude, period and observation index

    Parameters
    ----------
    K : array_like
        Coefficients of the vector of bottom friction, if K.size < number of volumes, interpolation
    A : float, optional
        Value of the amplitude used to compute the RSS, by default swe.amplitude = 5.0
    P : float, optional
        Value of the period used to compute the RSS, by default swe.amplitude = 15.0
    idx_to_observe : array_like
        Indices of the spatial domain where h is observed, by default idx_to_observe

    Returns
    -------
    tuple
        first element: tuple is the RSS, second gradient wrt to K.

    """
    Kcoeff = np.asarray(Kcoeff)
    K_transform = interp(Kcoeff)
    K = np.fromiter(map(K_transform, xr), dtype=float)
    cost, grad = J_grad_observed(K, A, P, idx_to_observe, hreference)
    grad_sum_length = int(xr.shape[0] / Kcoeff.size)
    grad_coeff = np.zeros(Kcoeff.size)
    for i in range(Kcoeff.size):
        grad_coeff[i] = sum(
            grad[i * grad_sum_length : (i * grad_sum_length + grad_sum_length)]
        )
    return cost, grad_coeff


# ------------------------------------------------------------------------------
def J_KAP_nograd(
    Kcoeff,
    A=amplitude,
    P=period,
    idx_to_observe=np.arange(href.shape[0]),
    hreference=href,
):
    """
    Computes RSS and its gradient wrt to K, with amplitude, period and observation index

    Parameters
    ----------
    K : array_like
        Coefficients of the vector of bottom friction, if K.size < number of volumes, interpolation
    A : float, optional
        Value of the amplitude used to compute the RSS, by default swe.amplitude = 5.0
    P : float, optional
        Value of the period used to compute the RSS, by default swe.amplitude = 15.0
    idx_to_observe : array_like
        Indices of the spatial domain where h is observed, by default idx_to_observe

    Returns
    -------
    tuple
        first element: tuple is the RSS, second gradient wrt to K.

    """
    Kcoeff = np.asarray(Kcoeff)
    K_transform = interp(Kcoeff)
    K = np.fromiter(map(K_transform, xr), dtype=float)
    bcL = lambda h, hu, t: BCrand(h, hu, t, "L", mean_h, A, P, phase)
    _, hcomputed, _, _ = diradj.shallow_water(
        D, g, T, h0, u0, N, LF_flux, dt, b, K, bcL, bcR
    )
    # print 'href: ', hreference
    # print 'hcomp: ', hcomputed
    # print 'idx: ', idx_to_observe
    cost, _ = J_function_observation(hcomputed, hreference, idx_to_observe)
    return cost


# ------------------------------------------------------------------------------
def J_piecewise_grad(Kcoeff, hreference=href):
    """Piecewise constant interpolation of K, then computes cost function and gradient"""
    K_transform = interp(Kcoeff)
    K = np.fromiter(map(K_transform, xr), dtype=float)
    cost, grad = J_grad(K)
    grad_sum_length = int(xr.shape[0] / Kcoeff.size)
    grad_coeff = np.zeros(Kcoeff.size)
    for i in range(Kcoeff.size):
        grad_coeff[i] = sum(
            grad[i * grad_sum_length : (i * grad_sum_length + grad_sum_length)]
        )
    return cost, grad_coeff


# ------------------------------------------------------------------------------
def J_pw_grad_complete(Kcoeff):
    K_transform = interp(Kcoeff)
    K = np.fromiter(map(K_transform, xr), dtype=float)
    cost, grad = J_grad(K)
    return cost, grad


# ------------------------------------------------------------------------------
def join_K_variables(design):
    npts, ndim = design.shape
    Kvar = design[:, :-2]
    if Kvar.shape[1] == 1:
        return design
    else:
        newdesign = [
            tuple([list(de[:-2]), de[-2], de[-1]]) for i, de in enumerate(design)
        ]
        return newdesign


# ------------------------------------------------------------------------------
def J_KAP_array(
    KAP,
    idx_to_observe=idx_to_observe,
    hreference=href,
    parallel=False,
    ncores=None,
    adj_gradient=False,
):
    """
    Computes RSS based on KAP as array and observation index,

    Parameters
    ----------
    KAP : (Kcoeff, A, P)
    Kcoeff : array_like
        interpolation
    A : float
        Value of the amplitude used to compute the RSS, by default swe.amplitude = 5.0
    P : float
        Value of the period used to compute the RSS, by default swe.amplitude = 15.0
    idx_to_observe : array_like
        Indices of the spatial domain where h is observed, by default idx_to_observe

    Returns
    -------
    tuple
        first element: tuple is the RSS, second gradient wrt to K.

    """
    if adj_gradient:
        fun_to_evaluate = J_KAP
    else:
        fun_to_evaluate = J_KAP_nograd
    KAP = np.asarray(join_K_variables(np.atleast_2d(KAP)))
    npoints = KAP.shape[0]
    trigger = True  # True for unparallelized
    if parallel:
        trigger = False
        if ncores is None:
            ncores = cpu_count()
        if npoints < 10:
            print("Not enough points to compute, switch to unparallelized proc")
            trigger = True

    if trigger:  # Unparallelized computations
        response = np.empty(npoints)
        start = time.time()
        if isinstance(KAP[0, 0], float):
            gradient = np.empty([npoints, 1])
        else:
            gradient = np.empty([npoints, len(KAP[0, 0])])

        for i, points in enumerate(KAP):
            if isinstance(points[0], float):
                arg_K = [points[0]]
            else:
                arg_K = points[0]

            if adj_gradient:
                response[i], gradient[i, :] = fun_to_evaluate(
                    np.asarray(arg_K),
                    points[1],
                    points[2],
                    hreference=hreference,
                    idx_to_observe=idx_to_observe,
                )
            else:
                response[i] = fun_to_evaluate(
                    np.asarray(arg_K),
                    points[1],
                    points[2],
                    hreference=hreference,
                    idx_to_observe=idx_to_observe,
                )
            print("Time elapsed for unparallelized computations", time.time() - start)

    else:  # Parallelized computations
        pool = Pool(processes=ncores)
        split = np.array_split(KAP, ncores, 0)
        start = time.time()
        if adj_gradient:
            response, gradient = zip(
                *pool.map(
                    partial(
                        J_KAP_array,
                        hreference=hreference,
                        idx_to_observe=idx_to_observe,
                        parallel=False,
                        adj_gradient=adj_gradient,
                    ),
                    split,
                )
            )
        else:
            response = pool.map(
                partial(
                    J_KAP_array,
                    hreference=hreference,
                    idx_to_observe=idx_to_observe,
                    parallel=False,
                    adj_gradient=adj_gradient,
                ),
                split,
            )
            # array_split necessary ?
        print("Time elapsed for parallelized computations", time.time() - start)
        response = np.asarray([item for sublist in response for item in sublist])
        if adj_gradient:
            gradient = np.asarray([item for sublist in gradient for item in sublist])
        pool.close()
        pool.join()

    if adj_gradient:
        return response, gradient
    else:
        return response


# ------------------------------------------------------------------------------
def transformation_variable_to_unit(X, bounds):
    """Transform from [bounds[:, 0], bounds[:, 1]] to [0, 1]"""
    Y = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    return Y


# ------------------------------------------------------------------------------
def inv_transformation_variable_to_unit(Y, bounds):
    """Transform from [0, 1] to [bounds[:, 0], bounds[:, 1]]"""
    X = Y * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    return X


#
#
#
#


# -- On direct run of the script ------------------------------------------------
if __name__ == "__main__":
    [xr, href2, uref2, t] = swe_KAP(Kref * 2, 5.02, 15.0)
    animate_SWE(xr, [href, href2], b, D, ylim=[0, 30])


# -- EOF -----------------------------------------------------------------------
