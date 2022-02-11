#!/usr/bin/env python
# -*- coding: utf-8 -*-

import HR_config.wrapper as sweHR
import swe.animation_SWE


# Animation of SSH -------------------------------------------

[xr, h, u, t] = sweHR.swe_KAP(sweHR.Kref * 2, sweHR.amplitude - 1.0, sweHR.period + 1.0)
swe.animation_SWE.animate_SWE(xr, [sweHR.href, h], sweHR.b, sweHR.D, ylim=[0, 40])

# ADjoint gradient
sweHR.J_KAP(sweHR.Kref, sweHR.amplitude, sweHR.period)
cost0, gradient0 = sweHR.J_KAP([0], sweHR.amplitude, sweHR.period)
epsilon = 1e-8
cost_eps = sweHR.J_KAP_nograd([epsilon], sweHR.amplitude, sweHR.period)
gradient_finite_diff = (cost_eps - cost0) / epsilon
print(gradient_finite_diff, gradient0)

response, gradient = sweHR.J_KAP_array(
    [
        ([0.1, 0.2, 0.5], 5.0, 15.0),  # Example array to evaluate
        ([0.1, 0.2, 0.5], 5.1, 15.1),  # Dim K = 3
        ([0.1, 0.1, 0.1], 5.0, 15.2),
        ([0.1, 0.1, 0.4], 5.1, 15.0),
        ([0.2, 0.2, 0.5], 5.0, 15.1),
        ([0.2, 0.2, 0.5], 5.1, 15.2),
        ([0.6, 0.1, 0.7], 5.0, 15.0),
        ([0.2, 0.2, 0.5], 5.1, 15.1),
        ([0.2, 0.2, 0.5], 5.0, 15.2),
        ([0.1, 0.7, 0.5], 5.1, 15.0),
        ([0.2, 0.2, 0.2], 5.0, 15.1),
    ],
    idx_to_observe=None,
    hreference=sweHR.href,
    parallel=False,
    ncores=4,
    adj_gradient=True,
)
