# Future sales prediction using Linear Regression

from sklearn.linear_model import LinearRegression

 

X = [[1], [2], [3], [4], [5]]  # Month number

y = [10000, 20000, 30000, 40000, 50000]  # Sales in rupees

 

model = LinearRegression()

model.fit(X, y)

 

future_month = [[6]]

print("Months:", [v[0] for v in X])

print("Sales:", y)

print("Predicted Sales for Month 6: Rs.", round(model.predict(future_month)[0], 2))

print("R2 Score:", model.score(X, y))
