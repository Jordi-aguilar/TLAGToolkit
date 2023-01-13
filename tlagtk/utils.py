from pybaselines.polynomial import modpoly, imodpoly
from BaselineRemoval import BaselineRemoval
  
def imodpoly_2(data, x_data):
    return imodpoly(data=data, x_data=x_data, poly_order=2, max_iter=500, tol=1e-6)[0]

def imodpoly_5(data, x_data):
    return imodpoly(data=data, x_data=x_data, poly_order=5, max_iter=500, tol=1e-6)[0]

BASELINES_FUNCTIONS = {
    "None" : lambda x: x,
    "imodpoly_2" : imodpoly_2,
    "imodpoly_5" : imodpoly_5
}