#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def LF_flux(FL, FR, lmax, lmin, UR, UL, dt, dx):
    ctilde = dx / dt
    return 0.5 * (FL + FR) - 0.5 * ctilde * (UR - UL)


def LF_flux_TL(
    FL, FR, lmax, lmin, UR, UL, dt, dx, FL_TL, FR_TL, lmax_TL, lmin_TL, UR_TL, UL_TL
):
    ctilde = dx / dt
    return np.array([0.5, 0.5, -ctilde / 2, ctilde / 2])
