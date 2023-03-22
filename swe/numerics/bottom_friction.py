import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("..")
sys.path.append("../..")

from swe.numerics.interpolation_tools import interp


class BottomFriction:
    def __init__(self, coefficient, spatial_domain) -> None:
        self.D = spatial_domain.D
        self.xr = spatial_domain.xr
        self.coefficient = coefficient
        K_transform = interp(np.asarray(self.coefficient), self.D)
        self.Karray = np.fromiter(map(K_transform, self.xr), dtype=float)
        self.eta = 7.0 / 3.0

    def evaluate(self, h, q):
        fric_quad = -1 * self.Karray * q * np.fabs(q) * (h ** (-self.eta))
        return fric_quad

    def tangent_linear(self, h, q):
        dfricdK = -q * np.fabs(q) * (h ** (-self.eta))
        dfricdh = -self.eta * self.Karray * q * np.fabs(q) * (h ** (-self.eta - 1))
        dfricdq = 2 * self.Karray * np.sign(q) * q * (h ** (-self.eta))
        return dfricdK, dfricdh, dfricdq

    def __call__(self, h, q):
        return self.evaluate(h, q)


# dt * 2 * K * np.sign(q[:, i]) * q[:, i] * (h[:, i] ** (-eta)) * q_d[:, i])

#  * K
#             * (eta)
#             * q[:, i]
#             * (np.fabs(q[:, i]) * (h[:, i] ** (-eta - 1)) * h_d[:, i])
#         )

if __name__ == "__main__":
    Nvolumes = 100
    D = [0, 10]
    dx = np.fabs(np.diff(D)) / Nvolumes
    xr = np.linspace(D[0] + dx / 2.0, D[1] - dx / 2.0, Nvolumes)
    coeff = [0.1, 0.2, 0.3]
    manning = BottomFriction(coeff, D, xr)
    print(manning.Karray)
    h = np.ones_like(xr)
    hu = np.ones_like(xr) * 3
    print(f"{manning.evaluate(h[:, 0], hu[:, 0]).shape=}")
    print(manning(h[:, 0], hu[:, 0]))
