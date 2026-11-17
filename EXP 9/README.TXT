
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Sample dataset
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10, 12])

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
linear_pred = linear_model.predict(X)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
poly_pred = poly_model.predict(X_poly)

# Plot the results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, linear_pred, color='red', label='Linear Regression')
plt.plot(X, poly_pred, color='green', label='Polynomial Regression')

plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Linear vs Polynomial Regression')
plt.legend()
plt.show()

print("Linear Predictions:", linear_pred)
print("Polynomial Predictions:", poly_pred)
