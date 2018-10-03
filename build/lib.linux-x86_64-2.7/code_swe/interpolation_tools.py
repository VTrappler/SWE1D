#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np


# -- Piecewise constant building blocks ---------------------------------------
def piecewise_cst(x, cell_vol):
    if np.abs(x) > (cell_vol / 2.0):
        res = 0
    else:
        res = 1
    return res


# -- Piecewise linear building block ------------------------------------------
def piecewise_lin(x, cell_vol):
    if np.abs(x) > (cell_vol / 2.0):
        res = 0
    else:
        res = -(2.0 / cell_vol) * np.sign(x) * x + 1
    return res


# -- Interpolation summation --------------------------------------------------
def interp_base(x, cell_vol, coef_array, pts, fun):
    sum_fun = 0
    for i, coef in enumerate(coef_array):
        sum_fun = sum_fun + coef * fun(x - pts[i], cell_vol)
    return sum_fun


# -- Piecewise constant interpolation -----------------------------------------
def interp_cst(x, cell_vol, coef_array, pts):
    return interp_base(x, cell_vol, coef_array, pts, piecewise_cst)


# -- Piecewise linear interpolation -------------------------------------------
def interp_lin(x, cell_vol, coef_array, pts):
    return interp_base(x, cell_vol, coef_array, pts, piecewise_lin)




# EOF -------------------------------------------------------------------------
