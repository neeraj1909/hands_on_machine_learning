import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression



iris = datasets.load_iris()
print(iris.keys())

# pedal width
X = iris["data"][:, 3:]

# 1 if iris virginica, else 0
y = (iris["target"] == 2).astype(np.int64)

log_reg = LogisticRegression()
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1,1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:,1], "g-", label="Iris virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris virginica")
plt.show()


print(log_reg.predict([[1.7], [1.6], [1.5],[1.6]]))
