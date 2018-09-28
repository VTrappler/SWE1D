from numpy import fabs

def LF_flux (FL, FR, lmax, lmin, UR, UL,dt,dx):
    ctilde = dx/dt
    return 0.5 * (FL + FR) - 0.5 * ctilde * (UR - UL)

