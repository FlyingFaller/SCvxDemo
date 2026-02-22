import numpy as np
from labeled_views import LabeledArray, LabeledParameter, LabeledVariable, LabeledExpression
from labeled_dim_view import LabeledDimArray
import cvxpy as cvx
from dimmed_views import DimProp

# states = ['t', 'px', 'py', 'pz']
# scalars = [1, 2, 3, 4]
# data= np.random.randint(0, 10, (10, 4))

# x = LabeledDimArray(data, scalars, states, axis=-1)

# print(x.nondim)

# states = ['A', 'B', 'C']
# data = np.arange(9).reshape((3,3))
# x = LabeledArray(data)
# xparam = LabeledParameter((3,3), states)
# xvar = LabeledVariable((3,3), states)

# dx = LabeledExpression(xvar - xparam, states)
# print(dx.shape)
# print(dx['A'].shape)
# print((xvar-xparam)[:, 0].name())
# # print(cvx.norm(dx['A']))
# # print(dx.is_constant())


# x = np.arange(5*10).reshape((10, 5))
# # (a, b, c) = np.split(x, [0, 1, 2])
# print(np.split(x, [1, 3, 5], axis=-1))
# x = np.eye(5)
# y = LabeledArray(x, ['A', 'B', 'C', 'D', 'E'])
# print(y[:, ['A']])

x = np.arange(-6, 6).reshape((3, 4))
print(x)
# print(np.linalg.norm(x, ord=1))
# print(np.linalg.norm(x.T, ord=1))

# print(np.sum(np.abs(x), axis=1))
print(np.abs(x).sum())
print(cvx.square(x).value)