# Credit score classification using Decision Tree

from sklearn.tree import DecisionTreeClassifier

 

# Features: monthly income (thousands), loan amount (thousands)

X = [[25, 5], [15, 8], [50, 10], [20, 9],

     [60, 12], [45, 8], [18, 10], [55, 7]]

y = ['Good', 'Poor', 'Good', 'Poor', 'Good', 'Good', 'Poor', 'Good']

 

model = DecisionTreeClassifier(random_state=0)

model.fit(X, y)

 

sample = [[30, 6]]

print("Dataset:", X)

print("Credit Labels:", y)

print("Predicted Credit Score:", model.predict(sample)[0])

print("Training Accuracy:", model.score(X, y) * 100, "%")
