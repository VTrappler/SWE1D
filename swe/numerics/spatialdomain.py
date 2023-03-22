import numpy as np
import matplotlib.pyplot as plt


class SpatialDomain1D:
    def __init__(self, D, Nvolumes) -> None:
        self.D = D
        self.Nvolumes = Nvolumes
        self.dx = (np.fabs(np.diff(D)) / self.Nvolumes).squeeze()
        self.xr = np.linspace(
            self.D[0] + self.dx / 2.0, self.D[1] - self.dx / 2.0, self.Nvolumes
        )
        self.bathymetry = np.zeros_like(self.xr)
        self.bathymetry_gradient = np.zeros_like(self.xr)

    def set_bathymetry(self, bathy_fun):
        self.bathymetry = bathy_fun(self.xr)
        self.bathymetry_gradient[1:-1] = (
            self.bathymetry[2:] - self.bathymetry[:-2]
        ) / (2 * self.dx)
        self.bathymetry_gradient[0] = (
            self.bathymetry[1] - self.bathymetry[0]
        ) / self.dx
        self.bathymetry_gradient[-1] = (
            self.bathymetry[-1] - self.bathymetry[-2]
        ) / self.dx

    def plot(self, ax=plt.gcf().gca(), bottom_friction=None):
        ax.plot(self.xr, self.bathymetry, color="k")
        ax.fill_between(x=self.xr, y1=0, y2=self.bathymetry, color="k", alpha=0.5)
        if bottom_friction is not None:
            ax.scatter(self.xr, self.bathymetry, s=bottom_friction.Karray)


if __name__ == "__main__":
    spatial_domain = SpatialDomain1D([0, 100], Nvolumes=50)
    spatial_domain.set_bathymetry(lambda x: x / 50)
    spatial_domain.plot()
