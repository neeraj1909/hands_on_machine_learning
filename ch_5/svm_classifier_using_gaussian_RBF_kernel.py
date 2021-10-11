from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC


X, y = make_moons(n_samples=100, noise=1.5)

rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
])

rbf_kernel_svm_clf.fit(X, y)

print(rbf_kernel_svm_clf.predict([[1.5, 0.0]]))
print(rbf_kernel_svm_clf.predict([[-1.0, 0.0]]))
