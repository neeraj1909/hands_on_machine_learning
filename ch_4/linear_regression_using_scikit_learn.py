import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # 4 + 3x + gaussian_noise

lin_reg = LinearRegression()
lin_reg.fit(X, y)

print("Linear Regression Coefficient:")
print(lin_reg.coef_)
print("Linear Regression Intercept:")
print(lin_reg.intercept_)
print("\n")

# Make prediction 
X_new = np.array([[0], [2]])
print("Priction for X:")
print(lin_reg.predict(X_new))


# np.c_ concatenate along 2nd axis
X_b = np.c_[np.ones((100, 1)), X]

# using scipy.linalg..lstsq()
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print("\nTheta based SVD:")
print(theta_best_svd)
print("Residuals:")
print(residuals)
print("Rank:")
print(rank)
print("s:")
print(s)
print("\n")

# Compute Pseudio-Inverse
print(np.linalg.pinv(X_b).dot(y)) 
