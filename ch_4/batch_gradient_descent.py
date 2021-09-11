import numpy as np

learning_rate = 0.1
no_of_iterations = 1000
m = 100

theta = np.random.randn(2, 1)   # random initialization

X = 2 *np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]   # adding x0 = 1 to each instance

"""
Formula for batch gradient descent:

MSE(theta) = 2/m * X_T * (X * theta - y)
"""

for iteration in range(no_of_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients
    
print(theta)
