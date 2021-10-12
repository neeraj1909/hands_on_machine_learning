from sklearn.svm import LinearSVR
from sklearn.datasets import make_moons


X, y = make_moons(n_samples=100, noise=1.5)
svm_reg = LinearSVR(epsilon=0.5)

svm_reg.fit(X, y)

print(svm_reg.predict([[1.0, 10.0]]))
print(svm_reg.predict([[0.0, 0.0]]))


'''
from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)
'''

'''
from sklearn.svm import SVR

svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)
'''
