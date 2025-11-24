import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# Set style for better looking plots
sns.set(style="whitegrid")

# Generate polynomial data
np.random.seed(42)
X = np.linspace(-3, 3, 100)
y = 0.5 * X**3 - 2 * X**2 + 1.5 * X + 2 + np.random.normal(0, 2, 100)
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Create a DataFrame for better visualization
data = pd.DataFrame({'X': X.flatten(), 'y': y.flatten()})

# Create polynomial features (degree=2 for quadratic, you can change this)
degree = 3  # Using degree 3 to match our cubic data generation
polynomial_features = PolynomialFeatures(degree=degree)

# Create a pipeline that first transforms the features and then fits the model
model = make_pipeline(polynomial_features, LinearRegression())
model.fit(X, y)

# Make predictions
X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_pred = model.predict(X_range)
y_pred_train = model.predict(X)

# Calculate metrics
mse = mean_squared_error(y, y_pred_train)
r2 = r2_score(y, y_pred_train)

print(f'Polynomial Regression (degree={degree})')
print(f'Mean squared error: {mse:.2f}')
print(f'R² score: {r2:.2f}')

# Create a figure with two subplots
plt.figure(figsize=(15, 6))

# First subplot: Polynomial regression fit
plt.subplot(1, 2, 1)
plt.scatter(data['X'], data['y'], color='black', label='Data points', alpha=0.7)
plt.plot(X_range, y_pred, color='green', linewidth=3, label=f'Polynomial (degree {degree})')
plt.title('Polynomial Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Second subplot: Residuals
residuals = y - y_pred_train
plt.subplot(1, 2, 2)
plt.scatter(y_pred_train, residuals, color='blue', alpha=0.7)
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
