#!/usr/bin/env python
# -*- coding: utf-8 -*-

import HR_config.wrapper as swe
import code_swe.animation_SWE


## Animation of SSH -------------------------------------------

[xr, h, u, t] = swe.swe_KAP(swe.Kref * 2, swe.amplitude - 1.0, swe.period + 1.0)
code_swe.animation_SWE.animate_SWE(xr, [swe.href, h], swe.b, swe.D, ylim = [0, 40])

## ADjoint gradient
swe.J_KAP(swe.Kref, swe.amplitude, swe.period)
cost0, gradient0 = swe.J_KAP([0], swe.amplitude, swe.period)
epsilon = 1e-8
cost_eps = swe.J_KAP_nograd([epsilon], swe.amplitude, swe.period)
gradient_finite_diff = (cost_eps - cost0) / epsilon
print gradient_finite_diff, gradient0

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
                                     parallel=False, ncores=4,
                                     adj_gradient=True)
