import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import log
from sklearn import linear_model

#Copied from https://github.com/mrandrewandrade/TER/blob/main/2025-11-19-Linear-Regression-Tutorial.md

regr_i = linear_model.LinearRegression()

# reads csv file and separates values into columns
anscombe_i = pd.read_csv('data/tests/anscombe_i.csv')

# make X and y in the shape sklearn expects
X = anscombe_i.x.to_numpy().reshape(-1, 1)
y = anscombe_i.y.to_numpy().reshape(-1, 1)

# fit the actual lr model
regr_i.fit(X, y)

# The coefficients
print('Coefficients: \n', regr_i.coef_)

# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr_i.predict(X) - y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr_i.score(X, y))

plt.plot(X,regr_i.predict(X), color='green',
         linewidth=3)

plt.scatter(anscombe_i.x, anscombe_i.y,  color='black')

plt.ylabel("X")
plt.xlabel("y")

plt.show()
