import numpy as np
import matplotlib.pyplot as plt
def locally_weighted_regression(test_point, X, y, tau):
 m = X.shape[0]
 weights = np.exp(-np.sum((X - test_point) ** 2, axis=1) / (2 * tau ** 2))
 W = np.diag(weights) 
 X_transpose = np.transpose(X)
 theta = np.linalg.inv(X_transpose @ W @ X) @ (X_transpose @ W @ y)
 return theta
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])
X_test = np.linspace(0, 5, 100)[:, np.newaxis]
tau_values = [0.1, 0.3, 1.0] # Bandwidth parameter values
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', label='Data')
for tau in tau_values:
 y_pred = [locally_weighted_regression(test_point, X, y, tau) for test_point in X_test]
 plt.plot(X_test, y_pred, label=f'Tau = {tau:.1f}')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Locally Weighted Regression')
plt.legend()
plt.show()
