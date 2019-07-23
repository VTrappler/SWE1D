#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


# ------------------------------------------------------------------------------
def J_function_observation(h, href, ind_to_observe=None):
    """Compute J, with the index of grid points observed"""
    space_dim = int(h.shape[0])
    if ind_to_observe is None:
        ind_to_observe = range(space_dim)
    obs_mat = np.zeros((href.shape[0], href.shape[0]))
    obs_mat[ind_to_observe, ind_to_observe] = 1
    J = 0.5 * np.mean(obs_mat.dot(h - href)**2)
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
    J = lambda h1, h2: 0.5 * np.mean(obs_mat.dot(h1 - h2)**2)
    return J, obs_mat


# EOF --------------
