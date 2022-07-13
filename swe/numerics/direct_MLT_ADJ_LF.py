#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# from LaxFriedrichs import LF_flux
from .compute_flux_1d import compute_flux_1d, compute_flux_1d_bis
from .variables import ConservedVars, PrimitiveVars
from .adjoint_function import ALFcons, BLFcons, CLFcons, DLFcons

from typing import Tuple, Union, Callable, List

g = 9.81
Initial = Union[Callable, np.ndarray]

# ------------------------------------------------------------------------------
def create_tridiag(sub: np.ndarray, diag: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Create a tridiagonal matrix

    :param sub: subdiagonal elements
    :type sub: np.ndarray
    :param diag: diagonal elements
    :type diag: np.ndarray
    :param upper: upperdiagonal elements
    :type upper: np.ndarray
    :return: Matrix constructed
    :rtype: np.ndarray
    """
    return np.diag(diag) + np.diag(sub, -1) + np.diag(upper, 1)


# ------------------------------------------------------------------------------
def DF(h: np.ndarray, u: np.ndarray, g: float) -> np.ndarray:
    return np.fabs(u) + np.sqrt(g * h)


# ------------------------------------------------------------------------------
def F(h: np.ndarray, u: np.ndarray, g: float) -> Tuple[np.ndarray, np.ndarray]:
    """Flux function for the SW

    :param h: Free surface height
    :type h: np.ndarray
    :param u: horizontal velocity
    :type u: np.ndarray
    :param g: gravitation constant
    :type g: float
    :return: Flux
    :rtype: tuple of np.ndarray
    """
    return (
        h * u,
        h * u * u + 0.5 * g * h * h,
    )


# ------------------------------------------------------------------------------
def shallow_water(
    D: List,
    g: float,
    T: float,
    h0: Initial,
    u0: Initial,
    N: int,
    num_flux: Callable,
    dt: float,
    b: Initial,
    Kvec: Union[List[float], np.ndarray],
    boundary_L: Callable,
    boundary_R: Callable,
    periodic: bool=False,
    external_forcing: Callable=None,
    tstart: float=0.0,
    verbose: bool=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a direct simulation of the SW
    """
    # Definition du pas, et initialisation des CI, et du vecteur des demi indices xr
    dx = np.fabs(np.diff(D)) / N
    xr = np.linspace(D[0] + dx / 2.0, D[1] - dx / 2.0, N)
    # x = np.linspace(D[0], D[1], N + 1)
    if callable(h0):
        h = h0(xr).squeeze()
    else:
        # if h0.squeeze().shape[0] != xr.shape[0]:
        #     pass # TODO: Add error
        h = h0
    if callable(u0):
        u = u0(xr).squeeze()
    else:
        # if h0.squeeze().shape[0] != xr.shape[0]:
        #     pass# TODO: Add error
        u = u0
    Nt = int((T - tstart) / dt + 1)
    h_array = np.zeros([xr.shape[0], Nt])
    h_array[:, 0] = h
    u_array = np.zeros([xr.shape[0], Nt])
    u_array[:, 0] = u
    t_array = np.empty(Nt)
    eta = 7.0 / 3.0
    if callable(Kvec):
        K = Kvec(xr)
        Kt = "function"
    else:
        K = Kvec
        Kt = "array"

    # Modification de la hauteur d'eau, si il y a une bathy non constante
    if b is not None:
        h = h - b(xr).squeeze()

    # Passage en var conservatives
    h, hu = ConservedVars(h, u)
    #  hu = h*u
    # gradient de la bathy
    if b is not None:
        B = b(xr)
        DB = (B[2:] - B[:-2]) / (2 * dx)
        DB0 = (B[1] - B[0]) / dx
        DBend = (B[-1] - B[-2]) / dx
        DB = np.insert(DB, 0, [DB0])
        DB = np.append(DB, [DBend])
        # maxB = np.fmax(B[:-1],B[1:])

    t = tstart
    i = 1
    if verbose:
        print("Debut de la simulation")
        print("K  = {}".format(Kt))
        print("Nt = {}".format(Nt))
        print("Nx = {}".format(N))
    # while t < T:
    while i < Nt:
        # Calcul du flux numerique, et valeur propre max
        # [Fh,Fhu, lmax,lmin] = compute_flux_1d_bis(h, hu, F, DF, g, num_flux, dt, dx)
        Fh, Fhu, lmax, lmin = compute_flux_1d(
            h, hu, F, DF, g, num_flux, dt, dx, periodic
        )

        # Adaptation du pas de temps, avec condition CFL
        # dt = min (T-t, CFL * dx/lmax)

        # Terme source
        if b is not None:
            S = -g * h * DB
        # print(f"{type(K)=}")
        # print(f"{type(hu)=}")
        # print(f"{type(np.fabs(hu))=}")
        # print(f"{type(h**(-eta))=}")
        fric_quad = -1 * K * hu * np.fabs(hu) * (h ** (-eta))
        # maj des variables d'etat conservatives

        h -= dt / dx * np.diff(Fh)
        hu -= dt / dx * np.diff(Fhu)
        if b is not None:
            # dd = g*(h[2:]**2 - h[:-2]**2)/(2.*dx)
            # dd = np.insert(dd,0,[0])
            # S = np.append(dd,0)
            hu += dt * S
        hu += dt * fric_quad

        # Conditions aux limites
        if not periodic:
            h, hu = boundary_L(h, hu, t)
            h, hu = boundary_R(h, hu, t)
        else:
            h, hu = boundary_L(h, hu, t)

        if external_forcing is not None:
            h, hu = external_forcing(h, hu, t)
        try:
            h_array[:, i], u_array[:, i] = PrimitiveVars(h, hu)
            t_array[i] = t
        except IndexError:
            print("IndexError thrown")
            print("i: {}, t: {}, T: {}".format(i, t, T))
            print("Nt: {}".format(Nt))

        if np.all(np.isnan(h)):
            print("Computation gives NaN(s)")
            break
        # Sauve pas de temps courant, et update i
        t += dt
        i = i + 1
    if verbose:
        print("Fin de la simulation")

    return xr, h_array, u_array, t_array


# ------------------------------------------------------------------------------
#                            Linear Tangent Model
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def lineaire_tangent_shallow_water(
    D: List,
    g: float,
    T: float,
    N: int,
    dt: float,
    b: Initial,
    Kvec: Union[List[float], np.ndarray],
    dK: Union[List[float], np.ndarray],
    h: np.ndarray,
    u: np.ndarray,
    href: np.ndarray,
    bcL_d: Callable,
    bcR_d: Callable,
    obs_mat: np.ndarray=None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    dx = np.fabs(np.diff(D)) / N
    xr = np.linspace(D[0] + dx / 2.0, D[1] - dx / 2.0, N)
    Nt = int(T / dt + 1)
    h_d = np.zeros([xr.shape[0], Nt])
    q_d = np.zeros([xr.shape[0], Nt])
    # t_d = np.zeros(Nt)
    eta = 7.0 / 3.0
    # lam = dx / dt

    if b is not None:
        B = b(xr)
        DZ = (B[2:] - B[:-2]) / (2 * dx)
        DZ0 = (B[1] - B[0]) / dx
        DZend = (B[-1] - B[-2]) / dx
        DZ = np.insert(DZ, 0, [DZ0])
        DZ = np.append(DZ, [DZend])
    else:
        DZ = np.zeros([xr.shape[0]])

    if callable(Kvec):
        K = Kvec(xr)
    else:
        K = Kvec * np.ones(xr.shape[0])
    i = 0
    t = 0
    q = h * u

    while i < Nt - 1:

        # dh_{t+1} = dh_t - c * (A @ dh_{t} + B @ dq_{t})
        # dq_{t+1} = dq_t - c * (C @ dh_{t} + D @ dq_{t})

        subA, diagA, supA = ALFcons(h[:, i], q[:, i], g, dt, dx, K)
        subB, diagB, supB = BLFcons(h[:, i], q[:, i], g, dt, dx, K)
        subC, diagC, supC = CLFcons(h[:, i], q[:, i], g, dt, dx, K, DZ)
        subD, diagD, supD = DLFcons(h[:, i], q[:, i], g, dt, dx, K)

        Amat = create_tridiag(subA, diagA, supA)
        Bmat = create_tridiag(subB, diagB, supB)
        Cmat = create_tridiag(subC, diagC, supC)
        Dmat = create_tridiag(subD, diagD, supD)

        h_d[:, i + 1] = h_d[:, i] - (dt / (2.0 * dx)) * (
            Amat.dot(h_d[:, i]) + Bmat.dot(q_d[:, i])
        )

        q_d[:, i + 1] = q_d[:, i] - (dt / (2.0 * dx)) * (
            Cmat.dot(h_d[:, i]) + Dmat.dot(q_d[:, i])
        )
        q_d[:, i + 1] += (
            -dt * 2 * K * np.sign(q[:, i]) * q[:, i] * (h[:, i] ** (-eta)) * q_d[:, i]
        )
        # Influence of bottom friction
        q_d[:, i + 1] += -dt * q[:, i] * np.fabs(q[:, i]) * (h[:, i] ** (-eta)) * dK
        q_d[:, i + 1] += (
            dt
            * K
            * (eta)
            * q[:, i]
            * (np.fabs(q[:, i]) * (h[:, i] ** (-eta - 1)) * h_d[:, i])
        )
        q_d[:, i + 1] += -dt * g * DZ * h_d[:, i]  # Influence of bottom topography

        h_d[:, i + 1], q_d[:, i + 1] = bcL_d(h_d[:, i + 1], q_d[:, i + 1], t)
        h_d[:, i + 1], q_d[:, i + 1] = bcR_d(h_d[:, i + 1], q_d[:, i + 1], t)

        t = t + dt
        i = i + 1

    if obs_mat is None:
        dj = np.sum((h - href) * h_d)  # scalar product
    else:
        dj = np.sum(obs_mat.T.dot(obs_mat.dot(h - href)) * h_d)
    # print("dj = ", dj)
    return h_d, q_d, dj


# ------------------------------------------------------------------------------
#                              Adjoint model
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def adjoint_shallow_water(
    D, g, T, N, dt, b, Kvec, h_array, u_array, ecartObs, bcL_A, bcR_A, obs_mat=None
):
    dx = np.fabs(np.diff(D)) / N
    xr = np.linspace(D[0] + dx / 2.0, D[1] - dx / 2.0, N)
    Nt = int(1 + T / dt)
    h = h_array
    u = u_array
    eta = 7.0 / 3.0
    lam = dx / dt
    q = h * u
    if obs_mat is None:
        obs_mat = np.diag(np.ones(ecartObs.shape[0]))

    if callable(Kvec):
        K = Kvec(xr)
    else:
        K = Kvec

    h_A = np.zeros([N, Nt])
    u_A = np.zeros([N, Nt])
    q_A = np.zeros([N, Nt])

    if b is not None:
        B = b(xr)
        DZ = (B[2:] - B[:-2]) / (2.0 * dx)
        DZ0 = (B[1] - B[0]) / dx
        DZend = (B[-1] - B[-2]) / dx
        DZ = np.insert(DZ, 0, [DZ0])
        DZ = np.append(DZ, [DZend])
    else:
        DZ = np.zeros([xr.shape[0]])
    t = T
    i = Nt - 2
    # print 'Debut resolution modele adjoint'
    while i > -1:

        subA, diagA, supA = ALFcons(h[:, i + 1], q[:, i + 1], g, dt, dx, K)
        subB, diagB, supB = BLFcons(h[:, i + 1], q[:, i + 1], g, dt, dx, K)
        subC, diagC, supC = CLFcons(h[:, i + 1], q[:, i + 1], g, dt, dx, K, DZ)
        subD, diagD, supD = DLFcons(h[:, i + 1], q[:, i + 1], g, dt, dx, K)

        Astar = create_tridiag(subA, diagA, supA).T
        Bstar = create_tridiag(subB, diagB, supB).T
        Cstar = create_tridiag(subC, diagC, supC).T
        Dstar = create_tridiag(subD, diagD, supD).T

        h_A[:, i + 1], q_A[:, i + 1] = bcL_A(h_A[:, i + 1], q_A[:, i + 1], t)
        h_A[:, i + 1], q_A[:, i + 1] = bcR_A(h_A[:, i + 1], q_A[:, i + 1], t)

        h_A[:, i] = (
            h_A[:, i + 1]
            - (dt / (2.0 * dx)) * (Astar.dot(h_A[:, i + 1]) + Cstar.dot(q_A[:, i + 1]))
            + dt * obs_mat.T.dot(obs_mat.dot(ecartObs[:, i + 1]))
            - dt * g * DZ * q_A[:, i + 1]
            + dt
            * K
            * (eta)
            * q[:, i + 1]
            * np.fabs(q[:, i + 1])
            * (h[:, i + 1] ** (-eta - 1))
            * q_A[:, i + 1]
        )

        q_A[:, i] = (
            q_A[:, i + 1]
            - (dt / (2.0 * dx)) * (Bstar.dot(h_A[:, i + 1]) + Dstar.dot(q_A[:, i + 1]))
            + dt
            * (-2 * K * np.sign(q[:, i + 1]) * q[:, i + 1] * h[:, i + 1] ** (-eta))
            * q_A[:, i + 1]
        )

        t = t - dt
        i = i - 1
    # print 'Fin du modele adjoint'
    if isinstance(K, (list, np.ndarray)):
        # grad = np.sum(-(h[:,:-1]**(-eta)) * q[:,:-1] * np.fabs(q[:,:-1]) * q_A,1)
        grad = -np.sum(q * np.fabs(q) * q_A * h ** (-eta), 1)

    else:
        # grad = np.sum(-(h[:,:-1]**(-eta)) * q[:,:-1] * np.fabs(q[:,:-1]) * q_A)
        grad = -np.sum(q * np.fabs(q) * q_A * h ** (-eta))

    return h_A, q_A, grad


# ------------------------------------------------------------------------------
def shallow_water_RSS(
    D, g, T, h0, u0, N, num_flux, dt, b, Kvec, bcL, bcR, href, cost_fun
):
    xr, h_array, u_array, t = shallow_water(
        D, g, T, h0, u0, N, num_flux, dt, b, Kvec, bcL, bcR
    )
    cost = cost_fun(h_array, href)
    print("J(K) = ", cost)
    return cost


# ------------------------------------------------------------------------------
def shallow_water_RSS_grad(
    D, g, T, h0, u0, N, num_flux, dt, b, Kvec, bcL, bcR, bcL_A, bcR_A, href, cost_fun
):
    # Modèle direct
    xr, h_array, u_array, t = shallow_water(
        D, g, T, h0, u0, N, num_flux, dt, b, Kvec, bcL, bcR
    )
    ecartObs = h_array - href
    # Modèle adjoint
    h_A, q_A, grad = adjoint_shallow_water(
        D, g, T, N, dt, b, Kvec, h_array, u_array, ecartObs, bcL_A, bcR_A
    )
    cost = cost_fun(h_array, href)
    print("J(K) = ", cost)
    print("||grad J||**2 =", np.sum(grad**2))
    return cost, grad


# ------------------------------------------------------------------------------
def shallow_water_RSS_grad_observation(
    D,
    g,
    T,
    h0,
    u0,
    N,
    num_flux,
    dt,
    b,
    Kvec,
    bcL,
    bcR,
    bcL_A,
    bcR_A,
    href,
    cost_fun,
    obs_mat=None,
):
    # Modèle direct
    [xr, h_array, u_array, t] = shallow_water(
        D, g, T, h0, u0, N, num_flux, dt, b, Kvec, bcL, bcR
    )
    ecartObs = h_array - href
    # Modèle adjoint
    [h_A, q_A, grad] = adjoint_shallow_water(
        D, g, T, N, dt, b, Kvec, h_array, u_array, ecartObs, bcL_A, bcR_A, obs_mat
    )
    cost = cost_fun(h_array, href)
    return [cost] + [grad]


####### Modele lineaire tangent + REGULARISATION  ################

# ------------------------------------------------------------------------------
def lineaire_tangent_shallow_water_regul(
    D, g, T, N, dt, b, Kvec, dK0, h, u, href, bcL_d, bcR_d, alpha, Kb
):
    dx = np.fabs(np.diff(D)) / N
    xr = np.linspace(D[0] + dx / 2, D[1] - dx / 2, N)
    Nt = T / dt + 1
    h_d = np.zeros([xr.shape[0], Nt])
    q_d = np.zeros([xr.shape[0], Nt])
    t_d = np.zeros(Nt)
    eta = 7.0 / 3.0
    lam = dx / dt

    if b is not None:
        B = b(xr)
        DZ = (B[2:] - B[:-2]) / (2 * dx)
        DZ0 = (B[1] - B[0]) / dx
        DZend = (B[-1] - B[-2]) / dx
        DZ = np.insert(DZ, 0, [DZ0])
        DZ = np.append(DZ, [DZend])
    else:
        DZ = np.zeros([xr.shape[0]])

    if callable(Kvec):
        K = Kvec(xr)
    else:
        K = Kvec * np.ones(xr.shape[0])
    dK = dK0 * np.ones(xr.shape[0])
    i = 0
    t = 0
    q = h * u
    while i < Nt - 1:

        [subA, diagA, supA] = ALFcons(h[:, i], q[:, i], g, dt, dx, K)
        [subB, diagB, supB] = BLFcons(h[:, i], q[:, i], g, dt, dx, K)
        [subC, diagC, supC] = CLFcons(h[:, i], q[:, i], g, dt, dx, K, DZ)
        [subD, diagD, supD] = DLFcons(h[:, i], q[:, i], g, dt, dx, K)

        Amat = create_tridiag(subA, diagA, supA)
        Bmat = create_tridiag(subB, diagB, supB)
        Cmat = create_tridiag(subC, diagC, supC)
        Dmat = create_tridiag(subD, diagD, supD)

        h_d[:, i + 1] = h_d[:, i] - (dt / (2.0 * dx)) * (
            Amat.dot(h_d[:, i]) + Bmat.dot(q_d[:, i])
        )

        q_d[:, i + 1] = (
            q_d[:, i]
            - (dt / (2.0 * dx)) * (Cmat.dot(h_d[:, i]) + Dmat.dot(q_d[:, i]))
            - dt * 2 * K * np.sign(q[:, i]) * q[:, i] * (h[:, i] ** (-eta)) * q_d[:, i]
            - dt * q[:, i] * np.fabs(q[:, i]) * (h[:, i] ** (-eta)) * dK
            + dt
            * K
            * (eta)
            * q[:, i]
            * np.fabs(q[:, i])
            * (h[:, i] ** (-eta - 1))
            * h_d[:, i]
            - dt * g * DZ * h_d[:, i]
        )

        [h_d[:, i + 1], q_d[:, i + 1]] = bcL_d(h_d[:, i + 1], q_d[:, i + 1], t)
        [h_d[:, i + 1], q_d[:, i + 1]] = bcR_d(h_d[:, i + 1], q_d[:, i + 1], t)

        t = t + dt
        i = i + 1
    dj = np.sum((h - href) * h_d) + alpha * (K - Kb) * dK
    print("dj = ", dj)
    return [h_d] + [q_d] + [dj]


# ------------------------------------------------------------------------------
def adjoint_regul(
    D, g, T, N, dt, b, Kvec, h_array, u_array, ecartObs, bcL_A, bcR_A, alpha, Kb
):
    dx = np.fabs(np.diff(D)) / N
    xr = np.linspace(D[0] + dx / 2, D[1] - dx / 2, N)
    Nt = int(1 + T / dt)
    h = h_array
    u = u_array
    eta = 7.0 / 3.0
    lam = dx / dt
    q = h * u

    if callable(Kvec):
        K = Kvec(xr)
    else:
        K = Kvec

    h_A = np.zeros([N, Nt])
    u_A = np.zeros([N, Nt])
    q_A = np.zeros([N, Nt])

    if b is not None:
        B = b(xr)
        DZ = (B[2:] - B[:-2]) / (2 * dx)
        DZ0 = (B[1] - B[0]) / dx
        DZend = (B[-1] - B[-2]) / dx
        DZ = np.insert(DZ, 0, [DZ0])
        DZ = np.append(DZ, [DZend])
    else:
        DZ = np.zeros([xr.shape[0]])
    t = T
    i = Nt - 2
    # print 'Debut resolution modele adjoint'
    while i > -1:

        [subA, diagA, supA] = ALFcons(h[:, i + 1], q[:, i + 1], g, dt, dx, K)
        [subB, diagB, supB] = BLFcons(h[:, i + 1], q[:, i + 1], g, dt, dx, K)
        [subC, diagC, supC] = CLFcons(h[:, i + 1], q[:, i + 1], g, dt, dx, K, DZ)
        [subD, diagD, supD] = DLFcons(h[:, i + 1], q[:, i + 1], g, dt, dx, K)

        Astar = create_tridiag(subA, diagA, supA).T
        Bstar = create_tridiag(subB, diagB, supB).T
        Cstar = create_tridiag(subC, diagC, supC).T
        Dstar = create_tridiag(subD, diagD, supD).T

        [h_A[:, i + 1], q_A[:, i + 1]] = bcL_A(h_A[:, i + 1], q_A[:, i + 1], t)
        [h_A[:, i + 1], q_A[:, i + 1]] = bcR_A(h_A[:, i + 1], q_A[:, i + 1], t)

        h_A[:, i] = (
            h_A[:, i + 1]
            - (dt / (2.0 * dx)) * (Astar.dot(h_A[:, i + 1]) + Cstar.dot(q_A[:, i + 1]))
            + dt * ecartObs[:, i + 1]
            - dt * g * DZ * q_A[:, i + 1]
            + dt
            * K
            * (eta)
            * q[:, i + 1]
            * np.fabs(q[:, i + 1])
            * (h[:, i + 1] ** (-eta - 1))
            * q_A[:, i + 1]
        )

        q_A[:, i] = (
            q_A[:, i + 1]
            - (dt / (2.0 * dx)) * (Bstar.dot(h_A[:, i + 1]) + Dstar.dot(q_A[:, i + 1]))
            + dt
            * (-2 * K * np.sign(q[:, i + 1]) * q[:, i + 1] * h[:, i + 1] ** (-eta))
            * q_A[:, i + 1]
        )

        t = t - dt
        i = i - 1
    # print 'Fin du modele adjoint'
    if isinstance(K, (list, np.ndarray)):
        # grad = np.sum(-(h[:,:-1]**(-eta)) * q[:,:-1] * np.fabs(q[:,:-1]) * q_A,1)
        grad = -np.sum(q * np.fabs(q) * q_A * h ** (-eta), 1)

    else:
        # grad = np.sum(-(h[:,:-1]**(-eta)) * q[:,:-1] * np.fabs(q[:,:-1]) * q_A)
        grad = -np.sum(q * np.fabs(q) * q_A * h ** (-eta)) + alpha * (K - Kb)

    return [h_A] + [q_A] + [grad]


# ------------------------------------------------------------------------------
def shallow_water_RSS_grad_regul(
    D,
    g,
    T,
    h0,
    u0,
    N,
    num_flux,
    dt,
    b,
    Kvec,
    bcL,
    bcR,
    bcL_A,
    bcR_A,
    href,
    cost_fun,
    alpha,
    Kb,
):
    # Modèle direct
    [xr, h_array, u_array, t] = shallow_water(
        D, g, T, h0, u0, N, num_flux, dt, b, Kvec, bcL, bcR
    )
    ecartObs = h_array - href

    # Modèle adjoint
    [h_A, q_A, grad] = adjoint_regul(
        D, g, T, N, dt, b, Kvec, h_array, u_array, ecartObs, bcL_A, bcR_A, alpha, Kb
    )
    cost = cost_fun(h_array, href)
    print("J(K) = ", cost)
    print("||grad J||**2 =", np.sum(grad**2))
    return [cost] + [grad]


# ------------------------------------------------------------------------------
def shallow_water_reconstruit(
    D, g, T, h0, u0, N, num_flux, dt, b, Kvec, boundary_L, boundary_R
):
    # TODO to finish eventually
    # Definition du pas, et initialisation des CI, et du vecteur des demi indices xr
    dx = np.fabs(np.diff(D)) / N
    xr = np.linspace(D[0] + dx / 2, D[1] - dx / 2, N)
    x = np.linspace(D[0], D[1], N + 1)
    h = h0(xr)
    u = u0(xr)
    Nt = T / dt + 1
    h_array = np.zeros([xr.shape[0], Nt])
    u_array = np.zeros([xr.shape[0], Nt])
    t_array = np.zeros(Nt)
    eta = 7.0 / 3.0
    if callable(Kvec):
        K = Kvec(xr)
        Kt = "function"
    else:
        K = Kvec
        Kt = "array"

    # Modification de la hauteur d'eau, si il y a une bathy non constante
    if b is not None:
        h = h - b(xr)

    # Passage en var conservatives
    [h, hu] = ConservedVars(h, u)

    # gradient de la bathy
    if b is not None:
        B = b(xr)
        DB = (B[2:] - B[:-2]) / (2 * dx)
        DB0 = (B[1] - B[0]) / dx
        DBend = (B[-1] - B[-2]) / dx
        DB = np.insert(DB, 0, [DB[0]])
        DB = np.append(DB, [DB[-1]])
        maxB = np.fmax(B[:-1], B[1:])

    t = 0
    i = 0
    # print 'Debut de la simulation'
    # print 'K  = ', Kt
    # print 'Nt = ', Nt
    # print 'Nx = ', N
    while t < T:

        h_pL = np.fmax(h[:-1] + B[:-1] - maxB, np.zeros(N - 1))
        h_pR = np.fmax(h[1:] + B[1:] - maxB, np.zeros(N - 1))
        # q_pL = h_pL * u[:-1]
        # q_pR = h_pR * u[1:]
        # S_pL = 0.5*g*(h[:-1]**2 - h_pL**2)
        # S_mR = 0.5*g*(h[1:]**2 - h_pR**2)
        # Calcul du flux numerique, et valeur propre max
        [Fh, Fhu, lmax, lmin] = compute_flux_1d_bis(h, hu, F, DF, g, num_flux, dt, dx)

        # Adaptation du pas de temps, avec condition CFL
        # dt = min (T-t, CFL * dx/lmax)

        # Terme source
        if b is not None:
            S = -g * h * DB

        # maj des variables d'etat conservatives
        for j in range(1, N - 1, 1):
            h[j] = h[j] - dt / dx * (Fh[j + 1] - Fh[j])
            if h[j] > 0:
                hu[j] = (
                    hu[j]
                    - dt / dx * (Fhu[j + 1] - Fhu[j])
                    + g
                    * dt
                    / (2 * dx)
                    * (h_pL[j] ** 2 - h[j - 1] ** 2 + h[j] ** 2 - h_pR[j] ** 2)
                )
            else:
                hu[j] = 0
        fric_quad = -K * hu * np.fabs(hu) * (h ** (-eta))

        hu = hu + dt * fric_quad
        t = t + dt

        # Conditions aux limites
        [h, hu] = boundary_L(h, hu, t)
        [h, hu] = boundary_R(h, hu, t)

        [h_array[:, i], u_array[:, i]] = PrimitiveVars(h, hu)

        # Sauve pas de temps courant, et update i
        t_array[i] = t
        i = i + 1
    print("Fin de la simulation")

    return [xr] + [h_array] + [u_array] + [t_array]
