# Compare KNN, Naive Bayes, Decision Tree and Logistic Regression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

 

X = [[1, 2], [2, 3], [2, 1], [3, 2],

     [7, 8], [8, 9], [8, 7], [9, 8]]

y = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']

 

models = {

    'KNN': KNeighborsClassifier(n_neighbors=3),

    'Naive Bayes': GaussianNB(),

    'Decision Tree': DecisionTreeClassifier(random_state=0),

    'Logistic Regression': LogisticRegression()

}

 

for name, model in models.items():

    model.fit(X, y)

    prediction = model.predict(X)

    print(name, "Accuracy:", accuracy_score(y, prediction) * 100, "%")
