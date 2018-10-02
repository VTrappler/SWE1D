#!/usr/bin/env python
#-*- coding: utf-8 -*-

# ------------------------------------------------------------------------------
#           Definition of variables in the adjoint code for different fluxes
# ------------------------------------------------------------------------------

import numpy as np

g = 9.81
DF = lambda h,u : np.fabs(u) + np.sqrt(g*h)

def lambdavec(hL, uL, hR, uR):
    lambL = DF(hL, uL)
    lambR = DF(hR, uR)
    return [lambL, lambR]

def lambdfun(hL, uL, hR, uR):
    temp = lambdavec(hL, uL, hR, uR)
    return [np.amax(temp, 0)] + [np.argmax(temp, 0)]

# np.argmax([lambL,lambR],0) # --> 1 si lambda_L>lambda_R, 1 sinon

# D = [0,100]
# N=200
# T=60
# dt =  0.01
# dx = np.diff(D) / float(N)
# xr = np.linspace(D[0] + dx/2, D[1] - dx/2 , N


# indices L,R
# calcul lamL,R (h[L,R],u[L,R]) ---> lamL,R
# calcul lambda = max() (lamL,lamR) --> lam
# calcul FluxL (hL,uL,g) --> FhL,FuL
# calcul FluxR (hR,uR,g) --> FhR,FuR
# calcul flux Rus (FhL,FhR,Fhul,FhuR,lambda,h,hu)
# update


# ------------------------------------------------------------------------------
#                    Variables Primitives Rusanov 
# ------------------------------------------------------------------------------

def subA(h,u,g,dt,dx):
    [lambda_array,LgR] = lambdfun(h[:-2],u[:-2],h[1:-1],u[1:-1]) # LgR = 1 if lambda_{i-1} > lambda_{i}
    subDiag = (u[:-2] - lambda_array\
               - LgR*(0.5*np.sqrt(g/h[:-2])*(h[:-2] - h[1:-1])))
    return subDiag*(-dt/(2*dx))

# ------------------------------------------------------------------------------
def diagA(h,u,g,dt,dx):
    [lambda_m1,LgRm1] = lambdfun(h[:-2],u[:-2],h[1:-1],u[1:-1])
    [lambda_p1,LgRp1] = lambdfun(h[1:-1],u[1:-1],h[2:],u[2:])
    RgLm1 = (np.ones(np.shape(LgRm1))-  LgRm1) # RgLm1 = 1 if lambda_{i} > lambda_{i-1}
    diag =lambda_m1 - RgLm1*(0.5*np.sqrt(g/h[1:-1])*(h[:-2] - h[1:-1]))\
           + lambda_p1 + LgRp1*(0.5*np.sqrt(g/h[1:-1])*(h[1:-1] - h[2:])) # LgRp1 = 1 if lambda_{i} > lambda_{i+1}
    return diag*(-dt/(2*dx))

# ------------------------------------------------------------------------------
def supA(h,u,g,dt,dx):
    [lambda_array,LgR] = lambdfun(h[1:-1],u[1:-1],h[2:],u[2:])
    RgL = (np.ones(np.shape(LgR))-  LgR) # RgL = 1 if lambda_{i+1} > lambda_i
    supDiag = u[2:] - lambda_array \
              - RgL*(0.5*np.sqrt(g/h[2:])*(h[1:-1] - h[2:]))
    return supDiag*(-dt/(2*dx))


# ------------------------------------------------------------------------------
def subB(h,u,g,dt,dx):
    [lambda_array,LgR] = lambdfun(h[:-2],u[:-2],h[1:-1],u[1:-1])
    subDiag = h[:-2] \
              - LgR*(np.sign(u[:-2])*(h[:-2] - h[1:-1])) # LgR = 1 if lambda_{i-1} > lambda_{i}
    return subDiag*(-dt/(2*dx))

# ------------------------------------------------------------------------------
def diagB(h,u,g,dt,dx):
    [lambda_m1,LgRm1] = lambdfun(h[:-2],u[:-2],h[1:-1],u[1:-1])
    [lambda_p1,LgRp1] = lambdfun(h[1:-1],u[1:-1],h[2:],u[2:])
    RgLm1 = (np.ones(np.shape(LgRm1))-  LgRm1) # RgLm1 = 1 if lambda_{i} > lambda_{i-1}
    diag = - RgLm1*(np.sign(u[1:-1])*(h[:-2] - h[1:-1]))\
           + LgRp1*(np.sign(u[1:-1]))*(h[1:-1] - h[2:]) # LgRp1 = 1 if lambda_{i} > lambda_{i+1}
    return diag*(-dt/(2*dx))

# ------------------------------------------------------------------------------
def supB(h,u,g,dt,dx):
    [lambda_array,LgR] = lambdfun(h[1:-1],u[1:-1],h[2:],u[2:])
    RgL = (np.ones(np.shape(LgR))-  LgR)    
    supDiag = h[2:]\
              - (np.ones(np.shape(LgR))-  LgR)*(np.sign(u[2:]))*(h[1:-1] - h[2:]) # RgL = 1 if lambda_{i+1} > lambda_i
    return supDiag*(-dt/(2*dx))    

# ------------------------------------------------------------------------------
def subC(h,u,g,dt,dx):
    [lambda_array,LgR] = lambdfun(h[:-2],u[:-2],h[1:-1],u[1:-1])
    subDiag = u[:-2]**2 - g*h[:-2] \
              - LgR*(0.5*np.sqrt(g/h[:-2])*(u[:-2] - u[1:-1])) # LgR = 1 if lambda_{i-1} > lambda_{i}
    return subDiag*(-dt/(2*dx))

# ------------------------------------------------------------------------------
def diagC(h,u,g,k,eta,DZ,dt,dx):
    [lambda_m1,LgRm1] = lambdfun(h[:-2],u[:-2],h[1:-1],u[1:-1])
    [lambda_p1,LgRp1] = lambdfun(h[1:-1],u[1:-1],h[2:],u[2:]) # LgRp1 = 1 if lambda_{i} > lambda_{i+1}
    RgLm1 = (np.ones(np.shape(LgRm1))-  LgRm1) # RgLm1 = 1 if lambda_{i} > lambda_{i-1}
    diag = -1*(-dt/(2*dx))*RgLm1*(0.5*np.sqrt(g/h[1:-1])*(u[:-2] - u[1:-1]))\
           +  LgRp1*(0.5*np.sqrt(g/h[1:-1])*(u[1:-1] - u[2:])) -2*dx*g*DZ[1:-1]\
           - 2*dx*k*(eta+2)*u[1:-1]*np.fabs(u[1:-1])*h[1:-1]**(-eta-3)
    return diag #  

# ------------------------------------------------------------------------------
def supC(h,u,g):
    [lambda_array,LgR] = lambdfun(h[1:-1],u[1:-1],h[2:],u[2:])
    RgL = (np.ones(np.shape(LgR))-  LgR)    
    supDiag = u[2:]**2 - g*h[2:] \
              - RgL*(0.5*np.sqrt(g/h[2:])*(u[1:-1] - u[2:])) # RgL = 1 if lambda_{i+1} > lambda_i
    return supDiag

# ------------------------------------------------------------------------------
def subD(h,u,g):
    [lambda_array,LgR] = lambdfun(h[:-2],u[:-2],h[1:-1],u[1:-1])
    subDiag = 2*u[:-2]*h[:-2] \
              - lambda_array - LgR*(np.sign(u[:-2]))*(u[:-2] - u[1:-1]) # LgR = 1 if lambda_{i-1} > lambda_{i}
    return subDiag    

# ------------------------------------------------------------------------------
def diagD(h,u,g,k,eta,dx):
    [lambda_m1,LgRm1] = lambdfun(h[:-2],u[:-2],h[1:-1],u[1:-1])
    [lambda_p1,LgRp1] = lambdfun(h[1:-1],u[1:-1],h[2:],u[2:]) #  LgRp1 = 1 if lambda_{i} > lambda_{i+1}
    RgLm1 = (np.ones(np.shape(LgRm1))-  LgRm1) # RgLm1 = 1 if lambda_{i} > lambda_{i-1}
    diag = lambda_m1 \
           - RgLm1*(np.sign(u[1:-1])*(u[:-2] - u[1:-1])) \
           + lambda_p1 + LgRp1*(np.sign(u[1:-1])*(u[1:-1] - u[2:]))\
           - 2*dx*np.sign(u[1:-1])*u[1:-1]*h[1:-1]**(-eta-2)
    return diag

# ------------------------------------------------------------------------------
def supD(h,u,g):
    [lambda_array,LgR] = lambdfun(h[:-2],u[:-2],h[1:-1],u[1:-1])
    RgL = (np.ones(np.shape(LgR))-  LgR)    
    supDiag = 2*u[2:]*h[2:] + lambda_array \
              - RgL*(np.sign(u[2:]))*(u[1:-1] - u[2:]) # RgL = 1 if lambda_{i+1} > lambda_i
    return supDiag


# ------------------------------------------------------------------------------
#                    Variables Conservatives Rusanov 
# ------------------------------------------------------------------------------

def subAcons(h,q,g,dt,dx):
    [lambda_array,LgR] = lambdfun(h[:-2],u[:-2],h[1:-1],u[1:-1]) # LgR = 1 if lambda_{i-1} > lambda_{i}
    subDiag =  - lambda_array\
               - LgR*(0.5*np.sqrt(g/h[:-2]) - np.sign(q[:-2])/h[:-2]**2)*(h[:-2] - h[1:-1])
    return subDiag*(-dt/(2*dx))

# ------------------------------------------------------------------------------
def diagAcons(h,q,g,dt,dx):
    [lambda_m1,LgRm1] = lambdfun(h[:-2],u[:-2],h[1:-1],u[1:-1])
    [lambda_p1,LgRp1] = lambdfun(h[1:-1],u[1:-1],h[2:],u[2:])
    RgLm1 = (np.ones(np.shape(LgRm1))-  LgRm1) # RgLm1 = 1 if lambda_{i} > lambda_{i-1}
    diag =lambda_m1 \
           - RgLm1*(0.5*np.sqrt(g/h[1:-1]) - np.sign(q[1:-1])/h[1:-1]**2)*(h[:-2] - h[1:-1])\
           + lambda_p1\
           + LgRp1*(0.5*np.sqrt(g/h[1:-1]) - np.sign(q[1:-1])/h[1:-1]**2)*(h[1:-1] - h[2:]) # LgRp1 = 1 if lambda_{i} > lambda_{i+1}
    return diag*(-dt/(2*dx))

# ------------------------------------------------------------------------------
def supAcons(h,q,g,dt,dx):
    [lambda_array,LgR] = lambdfun(h[1:-1],u[1:-1],h[2:],u[2:])
    RgL = (np.ones(np.shape(LgR))-  LgR) # RgL = 1 if lambda_{i+1} > lambda_i
    supDiag =  - lambda_array \
              - RgL*(0.5*np.sqrt(g/h[2:]) - np.sign(q[2:])/h[2:]**2 )*(h[1:-1] - h[2:])
    return supDiag*(-dt/(2*dx))


# ------------------------------------------------------------------------------
def subBcons(h,q,g,dt,dx):
    [lambda_array,LgR] = lambdfun(h[:-2],u[:-2],h[1:-1],u[1:-1])
    subDiag = 1 \
              - LgR*(np.sign(q[:-2])/h[:-2])*(h[:-2] - h[1:-1]) # LgR = 1 if lambda_{i-1} > lambda_{i}
    return subDiag*(-dt/(2*dx))

# ------------------------------------------------------------------------------
def diagB(h,u,g,dt,dx):
    [lambda_m1,LgRm1] = lambdfun(h[:-2],u[:-2],h[1:-1],u[1:-1])
    [lambda_p1,LgRp1] = lambdfun(h[1:-1],u[1:-1],h[2:],u[2:])
    RgLm1 = (np.ones(np.shape(LgRm1))-  LgRm1) # RgLm1 = 1 if lambda_{i} > lambda_{i-1}
    diag = - RgLm1*(np.sign(q[1:-1])/h[1:-1])*(h[:-2] - h[1:-1])\
           + LgRp1*(np.sign(q[1:-1])/h[1:-1])*(h[1:-1] - h[2:]) # LgRp1 = 1 if lambda_{i} > lambda_{i+1}
    return diag*(-dt/(2*dx))

# ------------------------------------------------------------------------------
def supB(h,u,g,dt,dx):
    [lambda_array,LgR] = lambdfun(h[1:-1],u[1:-1],h[2:],u[2:])
    RgL = (np.ones(np.shape(LgR))-  LgR)    
    supDiag = h[2:]\
              - (np.ones(np.shape(LgR))-  LgR)*(np.sign(u[2:]))*(h[1:-1] - h[2:]) # RgL = 1 if lambda_{i+1} > lambda_i
    return supDiag*(-dt/(2*dx))    

# ------------------------------------------------------------------------------
def subC(h,u,g,dt,dx):
    [lambda_array,LgR] = lambdfun(h[:-2],u[:-2],h[1:-1],u[1:-1])
    subDiag = u[:-2]**2 - g*h[:-2] \
              - LgR*(0.5*np.sqrt(g/h[:-2])*(u[:-2] - u[1:-1])) # LgR = 1 if lambda_{i-1} > lambda_{i}
    return subDiag*(-dt/(2*dx))

# ------------------------------------------------------------------------------
def diagC(h,u,g,k,eta,DZ,dt,dx):
    [lambda_m1,LgRm1] = lambdfun(h[:-2],u[:-2],h[1:-1],u[1:-1])
    [lambda_p1,LgRp1] = lambdfun(h[1:-1],u[1:-1],h[2:],u[2:]) # LgRp1 = 1 if lambda_{i} > lambda_{i+1}
    RgLm1 = (np.ones(np.shape(LgRm1))-  LgRm1) # RgLm1 = 1 if lambda_{i} > lambda_{i-1}
    diag = -1*(-dt/(2*dx))*RgLm1*(0.5*np.sqrt(g/h[1:-1])*(u[:-2] - u[1:-1]))\
           +  LgRp1*(0.5*np.sqrt(g/h[1:-1])*(u[1:-1] - u[2:])) -2*dx*g*DZ[1:-1] \
           - 2*dx*k*(eta+2)*u[1:-1]*np.fabs(u[1:-1])*h[1:-1]**(-eta-3)
    return diag #  

# ------------------------------------------------------------------------------
def supC(h,u,g):
    [lambda_array,LgR] = lambdfun(h[1:-1],u[1:-1],h[2:],u[2:])
    RgL = (np.ones(np.shape(LgR))-  LgR)    
    supDiag = u[2:]**2 - g*h[2:] \
              - RgL*(0.5*np.sqrt(g/h[2:])*(u[1:-1] - u[2:])) # RgL = 1 if lambda_{i+1} > lambda_i
    return supDiag

# ------------------------------------------------------------------------------
def subD(h,u,g):
    [lambda_array,LgR] = lambdfun(h[:-2],u[:-2],h[1:-1],u[1:-1])
    subDiag = 2*u[:-2]*h[:-2] \
              - lambda_array - LgR*(np.sign(u[:-2]))*(u[:-2] - u[1:-1]) # LgR = 1 if lambda_{i-1} > lambda_{i}
    return subDiag    

# ------------------------------------------------------------------------------
def diagD(h,u,g,k,eta,dx):
    [lambda_m1,LgRm1] = lambdfun(h[:-2],u[:-2],h[1:-1],u[1:-1])
    [lambda_p1,LgRp1] = lambdfun(h[1:-1],u[1:-1],h[2:],u[2:]) # LgRp1 = 1 if lambda_{i} > lambda_{i+1}
    RgLm1 = (np.ones(np.shape(LgRm1))-  LgRm1) # RgLm1 = 1 if lambda_{i} > lambda_{i-1}
    diag = lambda_m1 \
           - RgLm1*(np.sign(u[1:-1])*(u[:-2] - u[1:-1])) \
           + lambda_p1 + LgRp1*(np.sign(u[1:-1])*(u[1:-1] - u[2:]))\
           - 2*dx*np.sign(u[1:-1])*u[1:-1]*h[1:-1]**(-eta-2)
    return diag

# ------------------------------------------------------------------------------
def supD(h,u,g):
    [lambda_array,LgR] = lambdfun(h[:-2],u[:-2],h[1:-1],u[1:-1])
    RgL = (np.ones(np.shape(LgR))-  LgR)    
    supDiag = 2*u[2:]*h[2:] + lambda_array \
              - RgL*(np.sign(u[2:]))*(u[1:-1] - u[2:]) # RgL = 1 if lambda_{i+1} > lambda_i
    return supDiag
            

# ------------------------------------------------------------------------------
#             Variables Conservatives Lax-Friedrichs 
# ------------------------------------------------------------------------------

def ALFcons(h, q, g, dt, dx, k):
    """Construit la diagonale, la diag sup et la diag inf de A"""
    lam = dx / dt
    sub =  - lam * np.ones(q[:-1].shape)
    diag = 2 * lam * np.ones(q[:-2].shape)
    diag = np.insert(diag, 0, [lam]) # ajout pour BC
    diag = np.append(diag, [lam])
    sup = - lam * np.ones(q[:-1].shape) #q[:-2] ss BC 
    return [sub] + [diag] + [sup] 

def BLFcons(h,q,g,dt,dx,k):
    """Construit la diagonale, la diag sup et la diag inf de B"""
    lam = dx/dt
    sub = -np.ones(h[1:].shape)
    diag = np.zeros(h[2:].shape)
    diag = np.insert(diag,0,[-1])
    diag = np.append(diag,[1])
    sup = np.ones(h[1:].shape)
    return [sub] + [diag] + [sup] # 

def CLFcons(h,q,g,dt,dx,k,DZ):
    """Construit la diagonale, la diag sup et la diag inf de C"""
    lam = dx/dt
    eta = 7./3.
    u = q/h
    sub = u[:-2]**2  -g*h[:-2]
    sub = np.append(sub, [u[-2]**2 - g*h[-2]] )
    diag = np.zeros(h[:-2].shape)
    diag = np.insert(diag,0,[u[0]**2 - g*h[0]])
    diag = np.append(diag, [-u[-1]**2 + g*h[-1]])
    sup = -u[1:]**2 + g*h[1:]
    return [sub] + [diag] + [sup]

def DLFcons(h,q,g,dt,dx,k):
    """Construit la diagonale, la diag sup et la diag inf de D"""
    lam = dx/dt
    eta = 7./3.
    u = q/h
    sub = -2*u[:-1]- lam
    diag = 2*lam*np.ones(u[:-2].shape)
    diag = np.insert(diag, 0 , [- 2*u[0] - lam] )
    diag = np.append(diag,     [ 2*u[-1] + lam ] )
    sup = 2*u[1:] - lam
    return [sub] + [diag] + [sup]
