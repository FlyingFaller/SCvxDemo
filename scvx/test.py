import numpy as np
from labeled_views import LabeledArray, LabeledParameter, LabeledVariable
from labeled_dim_view import LabeledDimArray
import cvxpy as cvx
from dimmed_views import DimProp

states = ['t', 'px', 'py', 'pz']
scalars = [1, 2, 3, 4]
data= np.random.randint(0, 10, (10, 4))

x = LabeledDimArray(data, scalars, states, axis=-1)

print(x.nondim)
