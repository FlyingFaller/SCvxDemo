from dimensionalization import DimProp, NonDimProp
import numpy as np

m = DimProp(np.arange(9).reshape((3,3)), scalar = np.array([1, 2, 3]), axis=1)
print(m.ndim)