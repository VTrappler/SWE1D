#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# ------------------------------------------------------------------------------
#           Boundary conditions : direct, adj and MLT
# ------------------------------------------------------------------------------


class BoundaryCondition:
    def __init__(self, boundary_conditions_function, **kwargs) -> None:
        self.bc_function = boundary_conditions_function
        self.kwargs = kwargs

    def __call__(self, h, q, t):
        return self.bc_function(h, q, t, **self.kwargs)


class BoundaryConditionsSeq:
    def __init__(self, bc1, bc2) -> None:
        self.bc1 = bc1
        self.bc2 = bc2

    def apply(self, h, q, t):
        h1, q1 = self.bc1(h, q, t)
        return self.bc2(h1, q2, t)


# ------------------------------------------------------------------------------
def BC(h, hu, t, side):
    """Conditions aux limites du modele direct"""
    if side == "L":
        # h[0] = (4.01+3*np.sin(t*(2*np.pi)))*h[0]
        h[0] = 20 + 5 * (np.sin(t * (2 * np.pi) / 50))
        hu[0] = 0
    elif side == "R":
        h[-1] = h[-2]
        hu[-1] = 0.0
    return h, hu


# ------------------------------------------------------------------------------
def BC_MLT(h_d, q_d, side):
    """Conditions aux limites du modele lineaire tangent"""
    if side == "L":
        h_d[0] = 0
        q_d[0] = 0
    elif side == "R":
        h_d[-1] = h_d[-2]
        q_d[-1] = 0
    return h_d, q_d


# ------------------------------------------------------------------------------
def BC_ADJ(h_A, q_A, side):
    """Conditions aux limites du modele adjoint"""
    if side == "L":
        h_A[0] = 0.0
        q_A[0] = 0.0
    elif side == "R":
        h_A[-2] = h_A[-1] + h_A[-2]  # reflexion à droite
        h_A[-1] = 0.0
        q_A[-1] = 0.0
    return h_A, q_A


# ------------------------------------------------------------------------------
def BCrand(h, hu, t, side, mean_h, amplitude, period, phase):
    """Conditions aux limites du modele direct, avec plus de paramètres"""
    if side == "L":
        h[0] = mean_h + amplitude * np.sin((t * (2 * np.pi) / period) + phase)
        hu[0] = 0.0
    elif side == "R":
        h[-1] = h[-2]
        hu[-1] = hu[-2] * 0.0
    return h, hu


# ------------------------------------------------------------------------------
def BCsumsin(h, hu, t, side, mean_h, amplitude_vector, fundperiod, phase):
    """Conditions aux limites du modele direct, avec plus de paramètres"""
    if side == "L":
        h[0] = mean_h
        period = fundperiod
        for amp in amplitude_vector:
            h[0] += amp * np.sin((t * (2 * np.pi) / period) + phase)
            period /= 2.0
        hu[0] = 0.0
    elif side == "R":
        h[-1] = h[-2]
        hu[-1] = hu[-2] * 0.0
    return h, hu


# ------------------------------------------------------------------------------
def BCperiodic(h, hu, t):
    h[0], h[-1] = h[-1], h[0]
    hu[0], hu[-1], hu[-1], hu[0]
    return h, hu


bcL = lambda h, hu, t: BC(h, hu, t, "L")
bcL = BoundaryCondition(BC, side="L")
bcR = lambda h, hu, t: BC(h, hu, t, "R")
bcR = BoundaryCondition(BC, side="R")

bcL_d = lambda h, hu, t: BC_MLT(h, hu, "L")
bcL_d = BoundaryCondition(BC_MLT, side="L")

bcR_d = lambda h, hu, t: BC_MLT(h, hu, "R")
bcR_d = BoundaryCondition(BC_MLT, side="R")

bcL_A = lambda h, hu, t: BC_ADJ(h, hu, "L")
bcL_A = BoundaryCondition(BC_ADJ, side="L")

bcR_A = lambda h, hu, t: BC_ADJ(h, hu, "R")
bcR_A = BoundaryCondition(BC_ADJ, side="R")
