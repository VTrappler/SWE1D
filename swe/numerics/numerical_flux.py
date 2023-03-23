#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Tuple
import numpy as np

## Numerical Fluxes


@dataclass
class ShallowWaterState:
    h: np.ndarray
    u: np.ndarray
    q: np.ndarray = None
    g: float = 9.81

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


class NumericalFlux:
    def __init__(self, dt: float, dx: float) -> None:
        self.dx = dx
        self.dt = dt
        self.c_tilde = dx / dt

    def flux(self, flux_left, flux_right, var_left, var_right, lambda_max, lambda_min):
        pass

    def compute_flux_1d(self, sw_state):
        N_interfaces = sw_state.Nvolumes + 1
        right_indices = np.arange(N_interfaces)
        right_indices[-1] = sw_state.Nvolumes - 1
        left_indices = np.arange(-1, N_interfaces - 1)
        left_indices[0] = 0
        sw_state.u[left_indices] + np.sqrt(sw_state.h[left_indices] * sw_state.g)
        lambda_max_vec = np.fmax(
            sw_state.u[left_indices] + np.sqrt(sw_state.h[left_indices] * sw_state.g),
            sw_state.u[right_indices] + np.sqrt(sw_state.h[right_indices] * sw_state.g),
        )
        lambda_min_vec = np.fmin(
            sw_state.u[left_indices] + np.sqrt(sw_state.h[left_indices] * sw_state.g),
            sw_state.u[right_indices] + np.sqrt(sw_state.h[right_indices] * sw_state.g),
        )
        flux_h_left, flux_q_left = sw_state.flux_function(
            sw_state.h[left_indices], sw_state.u[left_indices]
        )
        flux_h_right, flux_q_right = sw_state.flux_function(
            sw_state.h[right_indices], sw_state.u[right_indices]
        )

        flux_h = self.flux(
            flux_left=flux_h_left,
            flux_right=flux_h_right,
            var_left=sw_state.h[left_indices],
            var_right=sw_state.h[right_indices],
            lambda_max=lambda_max_vec,
            lambda_min=lambda_min_vec,
        )

        flux_q = self.flux(
            flux_left=flux_q_left,
            flux_right=flux_q_right,
            var_left=sw_state.q[left_indices],
            var_right=sw_state.q[right_indices],
            lambda_max=lambda_max_vec,
            lambda_min=lambda_min_vec,
        )

        return flux_h, flux_q, np.max(lambda_max_vec), np.max(lambda_min_vec)


class LaxFriedrichFlux(NumericalFlux):
    def __init__(self, dt: float, dx: float) -> None:
        super().__init__(dt, dx)

    def flux(self, flux_left, flux_right, var_left, var_right, lambda_max, lambda_min):
        return 0.5 * (flux_left + flux_right) - 0.5 * self.c_tilde * (
            var_right - var_left
        )

    def __call__(
        self, flux_left, flux_right, var_left, var_right, lambda_max, lambda_min
    ):
        return self.flux(
            flux_left, flux_right, var_left, var_right, lambda_max, lambda_min
        )


def rusanov_flux(FL, FR, lmax, lmin, UR, UL, dt, dx):
    ctilde = max(np.fabs(lmax), np.fabs(lmin))
    return 0.5 * (FL + FR) - 0.5 * ctilde * (UR - UL)


def LF_flux(FL, FR, lmax, lmin, UR, UL, dt, dx):
    ctilde = dx / dt
    return 0.5 * (FL + FR) - 0.5 * ctilde * (UR - UL)


if __name__ == "__main__":
    import numpy as np

    h = np.linspace(20, 30, 50)
    u = np.linspace(2, 1, 50)
    sw_state = ShallowWaterState(h=h, u=u)
    lf_flux = LaxFriedrichFlux(1.0, 0.5)
    res = lf_flux.compute_flux_1d(sw_state)
