import numpy as np
import matplotlib.pyplot as plt

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # 4 + 3x + gaussian_noise


# Normal equation
# θ_cap = (X_T X)^−1 . X_T. y

 # np.c_ concatenate along 2nd axis
X_b = np.c_[np.ones((100, 1)), X]   # adding x0 = 1 to each instance
# print(X_b.shape)

# calculate theta using Normal equation
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("Theta computed by Normal Equation:")
print(theta_best)
print("\n")


# Make prediction using theta_best
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new] # adding x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)

print("Priction for X:")
print(y_predict)
print("\n")


# Plot the model prediction
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()
