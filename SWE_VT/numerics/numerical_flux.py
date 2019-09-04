#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import fabs

## Numerical Fluxes



def rusanov_flux(FL, FR, lmax, lmin, UR, UL, dt, dx):
    ctilde = max(fabs(lmax), fabs(lmin))
    return 0.5 * (FL + FR) - 0.5 * ctilde * (UR - UL)


def LF_flux(FL, FR, lmax, lmin, UR, UL, dt, dx):
    ctilde = dx / dt
    return 0.5 * (FL + FR) - 0.5 * ctilde * (UR - UL)




