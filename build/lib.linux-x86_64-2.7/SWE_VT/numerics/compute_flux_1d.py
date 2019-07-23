#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from variables import PrimitiveVars


# ------------------------------------------------------------------------------
def compute_flux_1d(h, hu, F, DF, g, num_flux, dt, dx):
    # Initialisation des variables
    N = h.shape[0] + 1
    Fh = np.zeros(N)
    Fhu = np.zeros(N)
    lmaxvec = np.zeros(N)
    lminvec = np.zeros(N)
    [h, u] = PrimitiveVars(h, hu)
    for i in xrange(0, N):
        # Valeur des indices droite ou gauche
        L = max(0, i - 1)
        R = min(i, N - 2)
        # DF = |u| + (g*h)**0.5

        lmaxvec[i] = max(u[L] + np.sqrt(np.max(h[L], 0) * g),
                         u[R] + np.sqrt(np.max(h[R], 0) * g))
        lminvec[i] = min(u[L] - np.sqrt(np.max(h[L], 0) * g),
                         u[R] + np.sqrt(np.max(h[R], 0) * g))
        # F fonction de flux (de l'equation initiale)
        [FhL, FhuL] = F(h[L], u[L], g)
        [FhR, FhuR] = F(h[R], u[R], g)

        Fh[i] = num_flux(FhL, FhR, lmaxvec[i], lminvec[i], h[R], h[L], dt, dx)
        Fhu[i] = num_flux(FhuL, FhuR, lmaxvec[i], lminvec[i], hu[R], hu[L], dt, dx)

    lmax = np.max(lmaxvec)
    lmin = np.min(lminvec)
    return [Fh, Fhu, lmax, lmin]



def compute_flux_1d_bis(h, hu, F, DF, g, num_flux, dt, dx):
    # Initialisation des variables
    N = h.shape[0] + 1
    Fh = np.zeros(N)
    Fhu = np.zeros(N)
    lmaxvec = np.zeros(N)
    lminvec = np.zeros(N)
    [h, u] = PrimitiveVars(h, hu)
    for i in xrange(0, N):
        # Valeur des indices droite ou gauche
        L = max(0, i - 1)
        R = min(i, N - 2)
        # DF = |u| + (g*h)**0.5
        lmaxvec[i] = np.fmax(u[L] + np.sqrt(h[L] * g), u[R] + np.sqrt(h[R] * g))
        lminvec[i] = np.fmin(u[L] - np.sqrt(h[L] * g), u[R] + np.sqrt(h[R] * g))
        # F fonction de flux (de l'equation initiale)
        [FhL, FhuL] = F(h[L], u[L], g)
        [FhR, FhuR] = F(h[R], u[R], g)

        Fh[i] = 0.5 * (FhL + FhR) - 0.5 * (dx / dt) * (h[R] - h[L])
        Fhu[i] = 0.5 * (FhuL + FhuR) - 0.5 * (dx / dt) * (hu[R] - hu[L])

    lmax = np.max(lmaxvec)
    lmin = np.min(lminvec)
    return [Fh, Fhu, lmax, lmin]


# EOF --------------------------------------------------------------------------
