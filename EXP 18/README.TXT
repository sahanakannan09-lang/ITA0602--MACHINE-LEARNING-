# Perceptron for binary Iris classification

from sklearn.linear_model import Perceptron

 

# Setosa = 0, Versicolor = 1

X = [

    [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2],

    [5.0, 3.4, 1.5, 0.2], [5.4, 3.9, 1.7, 0.4],

    [5.5, 2.3, 4.0, 1.3], [5.7, 2.8, 4.1, 1.3],

    [5.8, 2.7, 4.1, 1.0], [6.0, 2.7, 4.2, 1.3]

]

y = [0, 0, 0, 0, 1, 1, 1, 1]

 

model = Perceptron(max_iter=1000, random_state=0)

model.fit(X, y)

 

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)[0]

print("Predicted Iris Class:", "Setosa" if prediction == 0 else "Versicolor")

print("Training Accuracy:", model.score(X, y) * 100, "%")
