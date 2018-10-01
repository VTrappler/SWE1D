# SWE_VT

Numerical code modelling the 1D shallow water equations to build a toy problem of robust estimation of the bottom friction. Quadratic cost function is implemented between a reference simulation (twin experiments setting) and the simulation performed with given parameters. The gradient with respect to the bottom friction of this cost function is obtained via the adjoint model. In the directory haute_resolution, `wrapper_HR.py` contains readily available cost function(s)



## Installation
Clone the project and hope for the best

## Usage
### Animation of the evolution of the sea surface heights
```
import haute_resolution.wrapper_HR as swe
import code_swe.animation_SWE

[xr, h, u, t] = swe.swe_KAP(swe.Kref * 2, swe.amplitude - 1.0, swe.period + 1.0)

code_swe.animation_SWE.animate_SWE(xr, [swe.href, h], swe.b, swe.D, ylim = [0,10])
```

### Evaluation of the cost function (where `swe.href` is the simulation reference), and verification of the gradient

```
import haute_resolution.wrapper_HR as swe
swe.J_KAP(swe.Kref, swe.amplitude, swe.period)

cost0, gradient0 = swe.J_KAP([0], swe.amplitude, swe.period)

epsilon = 1e-8
cost_eps = swe.J_KAP_nograd([epsilon], swe.amplitude, swe.period)
gradient_finite_diff = (cost_eps - cost0) / epsilon

print gradient_finite_diff, gradient0
```

## Technical details
### Numerical scheme
Numerical solution computed via finite volume.
Adjoint code has been derived for reflexive boundary on the right, and Lax-Friedrich's flux inbetween the volumes.
