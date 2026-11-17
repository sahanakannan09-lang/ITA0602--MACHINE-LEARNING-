# Bank loan approval prediction using Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

 

# Features: income (thousands), credit history (1=Good, 0=Poor)

X = [[50, 1], [20, 0], [60, 1], [25, 0],

     [45, 1], [18, 0], [55, 1], [30, 1]]

y = ['Approved', 'Rejected', 'Approved', 'Rejected',

     'Approved', 'Rejected', 'Approved', 'Approved']

 

model = GaussianNB()

model.fit(X, y)

 

sample = [[50, 1]]

print("Applicant Details:", sample[0])

print("Predicted Loan Status:", model.predict(sample)[0])

print("Training Accuracy:", model.score(X, y) * 100, "%")
