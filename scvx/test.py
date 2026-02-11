import numpy as np
from named_arrays import NamedArray, NamedParameter, NamedVariable
import cvxpy as cvx

states = ['pos', 'vel', 'acc']
x = NamedArray(np.arange(27).reshape((3,3,3)), states, axis=1)
y = NamedVariable((3,3), states, axis=1)

# print(0*('test',) + ('other',))