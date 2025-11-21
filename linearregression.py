from sklearn.linear_model import LinearRegression
import numpy as np

# Example data
hours = np.array([[1], [2], [3], [4], [5]])   # X values
scores = np.array([55, 60, 70, 78, 85])      # y values

model = LinearRegression()   # make the model
model.fit(hours, scores)     # train the model

print(model.coef_)  # print the coefficients
print(model.intercept_)  # print the intercept

toPredict = int(input(": "))
print(model.predict([[toPredict]]))
