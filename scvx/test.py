import numpy as np
from named_views import NamedArray, NamedParameter, NamedVariable
from named_dim_view import NamedDimArray
import cvxpy as cvx
from dimmed_views import DimProp

states = ['t', 'px', 'py', 'pz']
scalars = [1, 2, 3, 4]
data= np.random.randint(0, 10, (10, 4))

x = NamedDimArray(data, scalars, states, axis=-1)

print(x.nondim)
