# EM algorithm using Gaussian Mixture Model

from sklearn.mixture import GaussianMixture

import numpy as np

 

X = np.array([[2], [3], [4], [10], [11], [12], [20], [21], [22]])

 

model = GaussianMixture(n_components=3, random_state=0)

model.fit(X)

labels = model.predict(X)

 

print("Dataset:", X.ravel().tolist())

for cluster in range(3):

    values = X[labels == cluster].ravel().tolist()

    print("Cluster", cluster + 1, ":", sorted(values))

print("EM model converged:", model.converged_)
