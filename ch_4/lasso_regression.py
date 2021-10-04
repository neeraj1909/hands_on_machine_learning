from sklearn.linear_model import Lasso, SGDRegressor

import numpy as np


m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

lasso_reg = Lasso(alpha=1)
lasso_reg.fit(X, y)
print(lasso_reg.predict([[1.5]]))

# using SGD (stochastic gradient descent)
sgd_reg = SGDRegressor(penalty="l1")
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict([[1.5]]))