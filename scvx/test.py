import numpy as np
from named_arrays import NamedArray, NamedParameter, NamedVariable
import cvxpy as cvx

states = ['A', 'B', 'C', 'D']
x = NamedArray(np.arange(81).reshape((3,3,3,3)), states, axis=-1)
y = NamedVariable((3, 4), states, axis=1)

# print(y[['A','C']])
print(x)
print(x[0, ...,'A'])

# y = np.arange(12).reshape((3,4))
# print(y[:, (0, 2)])