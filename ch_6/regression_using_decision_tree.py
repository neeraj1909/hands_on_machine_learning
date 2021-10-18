import numpy as np
from sklearn.tree import DecisionTreeRegressor


m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) ** 2
y = y + np.random.randn(m, 1) / 10

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)

print(tree_reg.predict([[0.6]]))
