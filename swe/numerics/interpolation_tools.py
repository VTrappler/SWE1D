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


def interp(coef_array, D, fun=interp_cst):
    """Interpolation using piecewiste constant values"""
    coef_array = np.array(coef_array)
    D_length = float(np.diff(D)[0])
    cell_vol = D_length / coef_array.size
    pts = np.linspace(
        cell_vol / 2.0,
        D_length - cell_vol / 2.0,
        num=coef_array.size,
    )
    f_to_ret = lambda x: fun(x, cell_vol, coef_array, pts)
    return f_to_ret


# EOF -------------------------------------------------------------------------
