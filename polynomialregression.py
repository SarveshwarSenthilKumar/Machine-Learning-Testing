import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def generate_sample_data(n_samples=100, noise=1.0, random_state=42):
    """Generate sample data with a polynomial relationship."""
    np.random.seed(random_state)
    X = np.sort(5 * np.random.rand(n_samples, 1), axis=0)
    y = np.sin(X).ravel() + np.random.normal(0, noise, X.shape[0])
    return X, y

def train_polynomial_regression(X, y, degree=2):
    """Train a polynomial regression model."""
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    return model, poly_features

def evaluate_model(model, X, y, poly_features):
    """Evaluate the model and return metrics."""
    X_poly = poly_features.transform(X)
    y_pred = model.predict(X_poly)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return y_pred

def plot_results(X, y, y_pred, degree):
    """Plot the results of polynomial regression."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual data', alpha=0.5)
    plt.plot(X, y_pred, color='red', label=f'Polynomial (degree {degree})')
    plt.title('Polynomial Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Generate sample data
    X, y = generate_sample_data(n_samples=100, noise=0.5)
    
    # Set polynomial degree (you can change this)
    degree = 3
    
    # Train the model
    model, poly_features = train_polynomial_regression(X, y, degree=degree)
    
    # Make predictions
    y_pred = evaluate_model(model, X, y, poly_features)
    
    # Plot results
    plot_results(X, y, y_pred, degree)
    
    # Print model coefficients
    print(f"\nPolynomial Coefficients (degree {degree}):")
    print(model.coef_)
    print(f"Intercept: {model.intercept_:.4f}")

if __name__ == "__main__":
    main()
