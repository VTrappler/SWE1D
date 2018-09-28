#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ANIMATION MODULE FOR SWE CODE -------------------------------------------

import numpy as np
import pylab
from matplotlib import animation
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
def def_lines(Nlines):
    return [pylab.plot([], [])[0] for _ in range(Nlines)]


# ------------------------------------------------------------------------------
def init_lines(lines):
    for line in lines:
        line.set_data([], [])
    return lines


# ------------------------------------------------------------------------------
def animate_curve(i, lines, h, b, xr):
    for j, line in enumerate(lines):
        line.set_data(xr, h[j][:, i] + b(xr))
    return lines


# ------------------------------------------------------------------------------
def animate_SWE(xr, h, b, D, ylim = [0, 10]):
    """
    Create an animation of the temporal evolution of the sea surface height 

    Parameters
    ----------
    xr: numpy array
        Spatial coordinates of the middle of the volumes
    h: array of numpy array
        array of the sea surface heights we want to display at once
    b: function
        function describing the bathymetry
    D: array
        horizontal limits of the considered domain
    ylim: array
        vertical limits of the displayed animation (default [0, 10])

    """
    Nlines = np.shape(h)[0]
    if b is None:
        b = lambda x: 0
    fig = pylab.figure()
    lines = def_lines(Nlines)
    pylab.xlim(D[0], D[1])
    pylab.ylim(ylim[0], ylim[1])
    animate = lambda i: animate_curve(i, lines, h, b, xr)
    init = lambda: init_lines(lines)
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=h[0].shape[1], blit=True,
                                  interval=1, repeat=False)
    pylab.fill_between(xr, b(xr), color = [0.5, 0.5, 0.5])
    pylab.xlabel('xr')
    pylab.ylabel('h')
    pylab.show()
    return ani


# ------------------------------------------------------------------------------
def snapshot(h, t=0, xr, b, D):
    plt.fill_between(xr, b(xr), color = [0.5, 0.5, 0.5])
    for j, hl in enumerate(h):
        plt.plot(hl[:, t] + b(xr))
    plt.show()


# h1 = np.zeros([4,5])
# h2 = np.zeros([4,5])

# xr = np.array([0,1,2,3])
# b = lambda x : -0.2
# D = [0,3]

# h= [h1,h2]
