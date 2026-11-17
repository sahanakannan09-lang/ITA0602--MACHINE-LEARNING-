# Iris classification using KNN; dataset values included

from sklearn.neighbors import KNeighborsClassifier

 

X = [

    [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2],

    [5.0, 3.4, 1.5, 0.2], [5.4, 3.9, 1.7, 0.4],

    [6.2, 3.4, 5.4, 2.3], [6.7, 3.1, 4.7, 1.5],

    [5.9, 3.0, 5.1, 1.8], [6.5, 3.0, 5.5, 1.8],

    [5.5, 2.3, 4.0, 1.3], [5.8, 2.7, 5.1, 1.9]

]

y = ['Setosa'] * 4 + ['Virginica'] * 4 + ['Versicolor'] * 2

 

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X, y)

sample = [[5.1, 3.5, 1.4, 0.2]]

 

print("New Flower Measurements:", sample[0])

print("Predicted Flower:", model.predict(sample)[0])

print("Training Accuracy:", model.score(X, y) * 100, "%")
