from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC


X, y = make_moons(n_samples=100, noise=1.5)

polynomial_svc_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge"))
])

polynomial_svc_clf.fit(X, y)

print(polynomial_svc_clf.predict([[1.5, 0.0]]))
print(polynomial_svc_clf.predict([[-1.0, 0.0]]))
