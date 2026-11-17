
from sklearn.neighbors import KNeighborsClassifier

# Training data
X = [[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]]
y = [0, 0, 0, 1, 1, 1]

# Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X, y)

# Test data
test_data = [[5, 5]]

# Prediction
prediction = model.predict(test_data)

print("K-Nearest Neighbours Algorithm")
print("Test Data:", test_data)
print("Predicted Class:", prediction[0])

