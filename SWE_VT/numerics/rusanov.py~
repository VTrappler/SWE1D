from numpy import fabs

def rusanov_flux (FL, FR, lmax, lmin, UR, UL,dt,dx):
    ctilde = max(fabs(lmax),fabs(lmin))
    return 0.5 * (FL + FR) - 0.5 * ctilde * (UR - UL)

