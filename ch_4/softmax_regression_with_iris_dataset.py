import numpy as np

from sklearn import datasets
from sklearn.linear_model import LogisticRegression



iris = datasets.load_iris()
print(iris.keys())

# pedal length, pedal width
X = iris["data"][:, (2, 3)]

# 1 if iris virginica, else 0
y = iris["target"]

softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10)
softmax_reg.fit(X, y)

print(softmax_reg.predict([[5, 2]]))
print(softmax_reg.predict_proba([[5, 2]]))
