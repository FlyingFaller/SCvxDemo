import numpy as np
from named_arrays import NamedArray, NamedParameter, NamedVariable
import cvxpy as cvx
from dimensionalization import DimProp

states = ['A', 'B', 'C', 'D']
x = NamedArray(np.arange(12).reshape((3,4)), states, axis=-1)
# y = NamedVariable((3, 4), states, axis=1)

x['A'] = [1, 2, 3]

print(x['B'])

# print(y[['A','C']])
# print(x)
# print(x[0, ...,'A'])

# y = np.arange(12).reshape((3,4))
# print(y[:, (0, 2)])

# z = DimProp(5, scalar=[2])
# print(z.ndim)