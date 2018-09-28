# -*- coding: utf-8 -*-
import numpy as np
import pylab
import scipy.optimize
#%%
def BC(h, hu, t, side):
    """ Conditions aux limites du modele direct"""
    if side =='L':
        # h[0] = (4.01+3*np.sin(t*(2*np.pi)))*h[0]
        h[0] = 20 + 5*(np.sin(t*(2*np.pi)/50))
        hu[0] = 0
    elif side =='R':
        h[-1] = h[-2]
        hu[-1] = hu[-2]*0
    return [h] + [hu] 

def BC_MLT(h_d, q_d, side):
    """ Conditions aux limites du modele lineaire tangent"""
    if side == 'L':
        h_d[0] = 0
        q_d[0] = 0
    elif side == 'R':
        h_d[-1] = h_d[-2]
        q_d[-1] = 0
    return [h_d] + [q_d]

def BC_ADJ(h_A, q_A, side):
    """ Conditions aux limites du modele adjoint"""
    if side == 'L':
        h_A[0] = 0
        q_A[0] = 0
    elif side == 'R':
        h_A[-2] = h_A[-1] + h_A[-2] # reflexion à droite
        h_A[-1] = 0
        q_A[-1] = 0
    return [h_A] + [q_A]

def BCrand(h, hu, t, side,mean_h,amplitude,period,phase):
    """ Conditions aux limites du modele direct, avec plus de paramètres"""
    if side =='L':
        # h[0] = (4.01+3*np.sin(t*(2*np.pi)))*h[0]
        h[0] = mean_h + amplitude*np.sin((t*(2*np.pi)/period) +phase)
        hu[0] = 0
    elif side =='R':
        h[-1] = h[-2]
        hu[-1] = hu[-2]*0
    return [h] + [hu] 


bcL = lambda h,hu,t : BC(h,hu,t,'L')
bcR = lambda h,hu,t : BC(h,hu,t,'R')
bcL_d = lambda h,q,t: BC_MLT(h,q,'L')
bcR_d = lambda h,q,t: BC_MLT(h,q,'R')
bcL_A = lambda h,q,t: BC_ADJ(h,q,'L')
bcR_A = lambda h,q,t: BC_ADJ(h,q,'R')