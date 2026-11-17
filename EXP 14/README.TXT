# House price prediction using Linear Regression

from sklearn.linear_model import LinearRegression

 

# Features: area (sq.ft), number of bedrooms

X = [[1000, 2], [1200, 2], [1500, 3], [1800, 3],

     [2000, 4], [2200, 4]]

y = [3000000, 3500000, 4500000, 5500000, 6000000, 6500000]

 

model = LinearRegression()

model.fit(X, y)

 

sample = [[1500, 3]]

print("House Dataset:", X)

print("Prices:", y)

print("Predicted House Price: Rs.", round(model.predict(sample)[0], 2))

print("R2 Score:", round(model.score(X, y), 3))
