from dataclasses import dataclass
import numpy as np
from typing import Callable, Union, List, Tuple
from numerics.variables import ConservedVars, PrimitiveVars
from numerics.boundary_conditions import BoundaryCondition, BoundaryConditionsSeq
from numerics.bottom_friction import BottomFriction
from numerics.spatialdomain import SpatialDomain1D
import matplotlib.pyplot as plt
from numerics.numerical_flux import LaxFriedrichFlux, NumericalFlux


class ShallowWaterState:
    def __init__(
        self, h: np.ndarray = None, u: np.ndarray = None, q: np.ndarray = None
    ) -> None:
        self._h = h
        self._u = u
        self._q = q

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, new_h):
        self._h = new_h
        self._q = new_h * self._u

    def __post_init__(self):
        self.Nvolumes = self.h.shape[0]
        if self.u is None and self.q is not None:
            self.u = self.q / self.h
        if self.q is None and self.u is not None:
            self.q = self.h * self.u

    def DF(self) -> np.ndarray:
        return np.fabs(self.u) + np.sqrt(self.g * self.h)

    def flux_function(self, h, u) -> Tuple[np.ndarray, np.ndarray]:
        return (h * u, h * u**2 + 0.5 * self.g * h**2)

    def conservative_vars(self):
        return self.h, self.q

    def primitive_vars(self):
        return self.h, self.u


def shallow_water(
    spatial_domain,
    T: float,
    dt: float,
    initial_conditions: dict,
    num_flux: NumericalFlux,
    bottom_friction: BottomFriction,
    boundary_conditions: BoundaryConditionsSeq,
    # external_forcing: Callable = None,
    tstart: float = 0.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a direct simulation of the SW
    """
    # Definition du pas, et initialisation des CI, et du vecteur des demi indices xr
    Nvol = spatial_domain.Nvolumes
    dx = spatial_domain.dx
    xr = spatial_domain.xr
    # x = np.linspace(D[0], D[1], N + 1)
    h = initial_conditions["h"]
    u = initial_conditions["u"]

    g = 9.81
    Nt = int((T - tstart) / dt + 1)
    h_array = np.zeros([Nvol, Nt])
    h_array[:, 0] = h
    u_array = np.zeros([Nvol, Nt])
    u_array[:, 0] = u
    t_array = np.empty(Nt)

    # Modification de la hauteur d'eau, si il y a une bathy non constante
    h = h - spatial_domain.bathymetry
    # Passage en var conservatives
    sw_state = ShallowWaterState(h, u)
    #  q = h*u
    # gradient de la bathy

    # maxB = np.fmax(B[:-1],B[1:])

    t = tstart
    i = 1
    # if verbose:
    #     print("Debut de la simulation")
    #     print("K  = {}".format(Kt))
    #     print("Nt = {}".format(Nt))
    #     print("Nx = {}".format(N))
    # while t < T:
    while i < Nt:

        # Adaptation du pas de temps, avec condition CFL
        # dt = min (T-t, CFL * dx/lmax)

        # Terme source
        bathymetry_source_term = -g * sw_state.h * spatial_domain.bathymetry_gradient
        quadratic_friction = bottom_friction.evaluate(sw_state.h, sw_state.q)

        # maj des variables d'etat conservatives

        # Calcul du flux numerique, et valeur propre max
        # [Fh,Fhu, lmax,lmin] = compute_flux_1d_bis(h, hu, F, DF, g, num_flux, dt, dx)
        # Fh, Fhu, lmax, lmin = compute_flux_1d(
        #     h, q, F, DF, g, num_flux, dt, dx, periodic
        # )
        # h -= dt / dx * np.diff(Fh)
        # q -= dt / dx * np.diff(Fhu)

        q = sw_state.q + bathymetry_source_term * dt + quadratic_friction * dt

        # h, q = boundary_conditions(h, q, t)

        # if external_forcing is not None:
        #     h, hu = external_forcing(h, hu, t)
        try:
            h_array[:, i], u_array[:, i] = PrimitiveVars(h, q)
            t_array[i] = t
        except IndexError:
            print("IndexError thrown")
            print("i: {}, t: {}, T: {}".format(i, t, T))
            print("Nt: {}".format(Nt))

        if np.all(np.isnan(h)):
            print("Computation gives NaN(s)")
            break
        # Sauve pas de temps courant, et update i
        t += dt
        i = i + 1
    if verbose:
        print("Fin de la simulation")

    return xr, h_array, u_array, t_array


if __name__ == "__main__":
    spatial_domain = SpatialDomain1D([0, 100], Nvolumes=50)
    spatial_domain.set_bathymetry(lambda x: x / 50)
    h0 = np.linspace(20, 21, spatial_domain.Nvolumes)
    u0 = np.zeros_like(h0)
    initial_conditions = {"h": h0, "u": u0}
    manning = BottomFriction(np.array([1, 5, 10]), spatial_domain)
    spatial_domain.plot(bottom_friction=manning)
    shallow_water(
        spatial_domain,
        10,
        dt=0.5,
        initial_conditions=initial_conditions,
        num_flux=None,
        bottom_friction=manning,
        boundary_conditions=None,
    )
