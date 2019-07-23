#!/usr/bin/env python
# -*- coding: utf-8 -*-



def HLL_flux(FL, FR, lmax, lmin, UR, UL):
    if lmin >= 0:
        Flux = FL
    elif (lmin < 0) and (lmax > 0):
        Flux = (lmax * FL - lmin * FR + lmax * lmin * (UR - UL)) / (lmax - lmin)
    elif (lmax <= 0):
        Flux = FR
    return Flux
