# SWE_VT

Numerical code modelling the 1D shallow water equations to build a toy problem of robust estimation of the bottom friction. Quadratic cost function is implemented between a reference simulation (twin experiments setting) and the simulation performed with given parameters. The gradient with respect to the bottom friction of this cost function is obtained via the adjoint model. In the module `\HR_config\`, `wrapper.py` contains readily available cost function(s)

Main project: vtrapple/These>


## Installation
This project is not (yet?) on the python package index (PyPI), hence has to be cloned from this gitlab repository (root access may be needed to install):
```
git clone git@gitlab.inria.fr:vtrapple/SWE_VT.git
cd SWE_VT
python setup.py install
```

## Examples of utilisation
### Animation of the evolution of the sea surface heights
```python
import HR_config.wrapper as swe
import code_swe.animation_SWE

[xr, h, u, t] = swe.swe_KAP(swe.Kref * 2, swe.amplitude - 1.0, swe.period + 1.0)
code_swe.animation_SWE.animate_SWE(xr, [swe.href, h], swe.b, swe.D, ylim = [0,10])
```

### Evaluation of the cost function (where `swe.href` is the simulation reference), and verification of the gradient

```python
import HR_config.wrapper as swe

swe.J_KAP(swe.Kref, swe.amplitude, swe.period)
cost0, gradient0 = swe.J_KAP([0], swe.amplitude, swe.period)
epsilon = 1e-8
cost_eps = swe.J_KAP_nograd([epsilon], swe.amplitude, swe.period)
gradient_finite_diff = (cost_eps - cost0) / epsilon
print gradient_finite_diff, gradient0
```

### Evaluation of the cost function parallelized with gradient
`J_KAP_array` is readily implemented, taking advantage of the Multiprocessing module of python in order to parallelize the computations. The function takes as input an array of tuples, each one in the following format: `(Coeff_K, Amplitude, Period)`, where `Coeff_K` is an array that will produce the piecewise constant interpolation on all the grid points, and `Amplitude` and `Period` are scalar that parametrize the left boundary condition.

```python
import HR_config.wrapper as swe

response, gradient = swe.J_KAP_array([([0.1, 0.2, 0.5], 5.0, 15.0),   # Example array to evaluate
                                      ([0.1, 0.2, 0.5], 5.1, 15.1),   # Dim K = 3
                                      ([0.1, 0.1, 0.1], 5.0, 15.2),
                                      ([0.1, 0.1, 0.4], 5.1, 15.0),
                                      ([0.2, 0.2, 0.5], 5.0, 15.1),
                                      ([0.2, 0.2, 0.5], 5.1, 15.2),
                                      ([0.6, 0.1, 0.7], 5.0, 15.0),
                                      ([0.2, 0.2, 0.5], 5.1, 15.1),
                                      ([0.2, 0.2, 0.5], 5.0, 15.2),
                                      ([0.1, 0.7, 0.5], 5.1, 15.0),
                                      ([0.2, 0.2, 0.2], 5.0, 15.1)],
                                      idx_to_observe = None,
                                      hreference = swe.href,
                                      parallel=True, ncores=4,
                                      adj_gradient=True)
```

## Technical details
### Numerical scheme
Numerical solution computed via finite volume.
Adjoint code has been derived for reflexive boundary on the right, and Lax-Friedrich's flux inbetween the volumes.

### Boundary condition on the left
The boundary condition on the left is parametrized as following:
```math
h(0,t) = \texttt{mean.h} + \texttt{amplitude} \cdot \sin\left(t \frac{2\pi}{\texttt{period}} + \texttt{phase}\right)
```


## 
