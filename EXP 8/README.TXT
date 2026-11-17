

from sklearn.linear_model import LinearRegression

# Training data
X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Test data
test_data = [[6]]

# Prediction
prediction = model.predict(test_data)

print("Linear Regression Algorithm")
print("Input Value:", test_data[0][0])
print("Predicted Value:", prediction[0])
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
