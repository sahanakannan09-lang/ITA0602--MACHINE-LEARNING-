# Car price prediction using Linear Regression

from sklearn.linear_model import LinearRegression

 

# Features: car age (years), mileage (thousand km)

X = [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [2, 15]]

y = [800000, 700000, 600000, 500000, 400000, 750000]

 

model = LinearRegression()

model.fit(X, y)

 

sample = [[2, 20]]

print("Dataset:", X)

print("Car Prices:", y)

print("Predicted Car Price: Rs.", round(model.predict(sample)[0], 2))

print("R2 Score:", round(model.score(X, y), 3))
