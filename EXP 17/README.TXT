# Mobile price range classification using Decision Tree

from sklearn.tree import DecisionTreeClassifier

 

# Features: RAM (GB), battery capacity (mAh)

X = [[2, 2000], [3, 2500], [4, 3000], [4, 3500],

     [6, 4000], [8, 5000], [6, 4500], [8, 4800]]

y = ['Low', 'Low', 'Medium', 'Medium',

     'High', 'High', 'High', 'High']

 

model = DecisionTreeClassifier(random_state=0)

model.fit(X, y)

 

sample = [[6, 4000]]

print("Mobile Features:", sample[0])

print("Predicted Price Range:", model.predict(sample)[0])

print("Training Accuracy:", model.score(X, y) * 100, "%")
