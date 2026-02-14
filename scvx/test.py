import numpy as np
from named_arrays import NamedArray, NamedParameter, NamedVariable
import cvxpy as cvx
from dimensionalization import DimProp

states = ['A', 'B', 'C', 'D']
data= np.arange(12).reshape((3, 4))
x = NamedArray(data, states, axis=-1)
print(x['A':'A']['B'])