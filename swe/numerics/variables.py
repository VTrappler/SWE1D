#!/usr/bin/env python
# -*- coding: utf-8 -*-


def ConservedVars(h, u):
    return h, h * u


def ConservedVars2D(h, u, v=None):
    return [h, h * u, h * v] if v is not None else [h, h * u]


def PrimitiveVars(h, hu):
    return h, hu / h


def PrimitiveVars2D(h, hu, hv=None):
    return [h, hu / h, hv / h] if hv is not None else [h, hu / h]
