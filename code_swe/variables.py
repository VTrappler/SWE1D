def ConservedVars(h,u,v=None):
    return [h,h*u,h*v] if v != None else [h, h*u]

def PrimitiveVars(h,hu,hv=None):
    return [h,hu/h,hv/h] if hv != None else [h, hu/h]
