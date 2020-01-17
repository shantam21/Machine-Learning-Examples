import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import helper_functions

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(fit_intercept=True, max_iter=100000)

data = pd.read_csv('diabetes.csv')

print(data.head())

X = data.iloc[:,0:8]
print(X.head())

y = data.iloc[:,-1]
print(y.head())

start_time = time.time()

num_iter = 10000

intercept = np.ones((X.shape[0], 1)) 
X = np.concatenate((intercept, X), axis=1)
theta = np.zeros(X.shape[1])
# X = X.astype(float)
for i in range(num_iter):
    h = helper_functions.sigmoid(X, theta)
    gradient = helper_functions.gradient_descent(X, h, y)
    theta = helper_functions.update_weight_loss(theta, 0.01, gradient)
    
print("Training time (Log Reg using Gradient descent):" + str(time.time() - start_time) + " seconds")
print("Learning rate: {}\nIteration: {}".format(0.1, num_iter))

result = helper_functions.sigmoid(X, theta)
f = pd.DataFrame(np.around(result, decimals=6)).join(y)
f['pred'] = f[0].apply(lambda x : 0 if x < 0.5 else 1)
print("Accuracy (Loss minimization):")
print(f.loc[f['pred']==f['Outcome']].shape[0] / f.shape[0] * 100)


