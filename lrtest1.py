import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import log
from sklearn import linear_model
import seaborn as sns

# Set style for better looking plots
sns.set(style="whitegrid")

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

# Calculate predictions first
predictions = regr_i.predict(X)
residuals = y - predictions

# Create a figure with two subplots
plt.figure(figsize=(15, 6))

# First subplot: Regression line with data points and residual lines
plt.subplot(1, 2, 1)
plt.scatter(anscombe_i.x, anscombe_i.y, color='black', label='Data points')
plt.plot(X, predictions, color='green', linewidth=3, label='Regression line')

# Add vertical lines showing residuals
for xi, yi, pred in zip(X, y, predictions):
    plt.plot([xi[0], xi[0]], [yi[0], pred[0]], 'b-', alpha=0.7, linewidth=1.5, zorder=1)
# Add a single line to the legend for residuals
plt.plot([], [], 'b-', alpha=0.7, linewidth=1.5, label='Residuals')

plt.title('Linear Regression Fit with Residuals')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Second subplot: Residual plot
plt.subplot(1, 2, 2)
plt.scatter(predictions, residuals, color='blue', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')

plt.tight_layout()
plt.show()

# Create a separate figure for the histogram of residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='purple', bins=15)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.axvline(x=0, color='red', linestyle='--')
plt.show()
