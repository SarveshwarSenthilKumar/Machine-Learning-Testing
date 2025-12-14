# Machine Learning Testing Repository

Welcome to my machine learning testing repository! This is a personal space where I experiment with various machine learning concepts, algorithms, and techniques. Feel free to explore the code and learn alongside me as I dive into the world of machine learning.

## 📁 Repository Contents

1. **linearregression.py**
   - A simple implementation of linear regression using scikit-learn
   - Predicts scores based on study hours using a small dataset
   - Includes model training, coefficient analysis, and prediction functionality

2. **lrtest1.py**
   - An advanced linear regression example using the Anscombe's quartet dataset
   - Features comprehensive visualization including:
     - Regression line with residuals
     - Residual plot
     - Distribution of residuals
   - Uses pandas for data manipulation and seaborn for enhanced visualizations

3. **polynomialregression.py**
   - Implements polynomial regression using scikit-learn
   - Features include:
     - Customizable polynomial degree
     - Model evaluation metrics (MSE, R² score)
     - Data visualization
     - Sample data generation with customizable noise

4. **data/tests/anscombe_i.csv**
   - Part of the Anscombe's quartet dataset
   - Used in lrtest1.py for regression analysis
   - Demonstrates important statistical properties of linear regression

4. **requirements.txt**
   - Lists the Python dependencies required to run the code
   - Includes packages like scikit-learn, numpy, pandas, matplotlib, and seaborn

## 🚀 Getting Started

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the Python scripts to see the machine learning models in action:
   ```
   # Run polynomial regression
   python polynomialregression.py
   ```
   
   The script will:
   - Generate sample data with a polynomial relationship
   - Train a polynomial regression model (default: 3rd degree)
   - Display evaluation metrics (MSE and R² score)
   - Show a plot of the regression line

   You can adjust the polynomial degree by modifying the `degree` parameter in the `main()` function.

## 🔍 About Me

Hi! I'm Sarveshwar Senthil Kumar, a machine learning enthusiast exploring the fascinating world of AI and data science. This repository serves as my personal playground for learning and experimenting with various ML concepts.

### Connect with me:
- 🌐 Portfolio: [sarveshwarsenthilkumar.github.io](https://sarveshwarsenthilkumar.github.io)
- 💻 GitHub: [github.com/sarveshwarsenthilkumar](https://github.com/sarveshwarsenthilkumar)

## 📝 Notes
- This is a work in progress, and I'll be adding more experiments and projects over time.
- Feel free to explore the code, open issues, or suggest improvements!

Happy coding! 🚀
