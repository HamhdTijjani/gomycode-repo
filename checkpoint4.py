# Regression
# This is a dataset containing over 21613 houses and their characteristics. The goal is to find the best model to predict a house’s price. 
# 1. Explore this dataset using what you have learned in data preprocessing and data visualization 
# 2. Write a paragraph selecting the most important features (feature selection). 
# 3. Split your dataset into a training set and a testing set. 
# 4. Apply linear regression to your training set. 
# 5. Plot the linear regression. 
# 5. Measure the performance of linear regression using the testing set. 
# 6. Apply multiple-linear regression and compare it to the linear model. 
# 7. Apply polynomial regression and compare it to linear and multilinear regression. 
# Note: Every result has to be interpreted and justified. Write your interpretations in a markdown.

# Import relevant library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn import metrics

# Load the dataset
data = pd.read_csv('Workspace/GoMyCode/kc_house_data.csv')
print(data.head())

# # Explore the dataset  

print(data.info())
print(data.shape)
print(data.describe())

# # Data visualization
sns.pairplot(data, x_vars=['bedrooms', 'bathrooms','sqft_living','sqft_lot'], y_vars='price', height=5, aspect=0.8)
plt.show()

# # Identify and Treating outliers
# # Calculate the IQR for data
bathQ1 = data['bathrooms'].quantile(0.25)
bathQ3 = data['bathrooms'].quantile(0.75)
bathIQR = bathQ3 - bathQ1
thresold = 1.5
bath_outliers = data[(data['bathrooms'] < bathQ1 - thresold * bathIQR) | (data['bathrooms'] > bathQ3 + thresold * bathIQR)]
# print(bath_outliers.shape)
bedQ1 = data['bedrooms'].quantile(0.25)
bedQ3 = data['bedrooms'].quantile(0.75)
bedIQR = bedQ3 - bedQ1
thresold = 1.5
bed_outliers = data[(data['bedrooms'] < bedQ1 - thresold * bedIQR) | (data['bedrooms'] > bedQ3 + thresold * bedIQR)]
# print(bed_outliers.shape)
sqft_livQ1 = data['sqft_living'].quantile(0.25)
sqft_livQ3 = data['sqft_living'].quantile(0.75)
sqft_livIQR = sqft_livQ3 - sqft_livQ1
thresold = 1.5
sqft_liv_outliers = data[(data['sqft_living'] < sqft_livQ1 - thresold * sqft_livIQR) | (data['sqft_living'] > sqft_livQ3 + thresold * sqft_livIQR)]
# print(sqft_liv_outliers.shape)
sqft_lotQ1 = data['sqft_lot'].quantile(0.25)
sqft_lotQ3 = data['sqft_lot'].quantile(0.75)
sqft_lotIQR = sqft_lotQ3 - sqft_lotQ1
thresold = 1.5
sqft_lot_outliers = data[(data['sqft_lot'] < sqft_lotQ1 - thresold * sqft_lotIQR) | (data['sqft_lot'] > sqft_lotQ3 + thresold * sqft_lotIQR)]
# print(sqft_lot_outliers.shape)

outliers = (bath_outliers + bed_outliers + sqft_liv_outliers + sqft_lot_outliers)
# print(outliers.shape)

# Excluding Outliers from our data
data = data.drop(outliers.index)
print(data.shape)

# # Convert the datetime formate
data['date'] = pd.to_datetime(data['date'])


# Checking for missing data
print(data.isnull().sum())

# # # Feature selection
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=None,cmap=None)
plt.show()
print(correlation_matrix)

# Data visualization
sns.pairplot(data, x_vars=['bedrooms', 'bathrooms','sqft_living','sqft_lot'], y_vars='price', height=5, aspect=0.8)
plt.show()

# Linear Regression
# Select features based on correlation
selected_features = ['sqft_living']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data[selected_features],
                                                    data['price'],
                                                    test_size=0.2,
                                                    random_state=42)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# # Initiating and Training the model
linear_model =LinearRegression()
linear_model.fit(X_train,y_train)
y_pred = linear_model.predict(X_test)


# # Accuracy
MSE = mean_squared_error(y_test,y_pred)
R2 = metrics.r2_score(y_test,y_pred)
print(f'Linear Regression Mean Squared Error is {MSE}')
print(f'Linear Regression R squared is {R2}')
print(f'Linear Regression Coefficient is {linear_model.coef_}')
print(f'Linear Regression Intercept is {linear_model.intercept_}')

""" Interpretation:
The MSE of approximately 47.5 billion indicates the average squared difference between the predicted house prices and the actual prices.
An R² of approximately 0.37 suggests that around 37% of the variance in house prices is explained by the selected features in the model.
While this is not a high value, it indicates some level of predictive power
The coefficient of approximately 225 suggests that for every unit increase in the selected feature, the predicted house price increases by $225. 
The intercept represents the value of the dependent variable when all independent variables are zero. 
Here, it indicates that the predicted house price is approximately $58,767 when all features are zero.
while the linear regression model provides some predictive power, the relatively high MSE and moderate R² suggest that it may not capture the full complexity of the relationship between features and house prices
"""
# Linear Regression Plot
plt.scatter(X_test,y_test, color='r')
plt.plot(X_test,y_pred,color='k')
plt.title("Linear Regression")
plt.ylabel('Price')
plt.xlabel('Square Feet Living')
plt.show()

# Multilinear Regression
# Select features based on correlation
x  = data.drop(['id','date','price'], axis=1)
y = data['price']

# # Split data
X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(x,
                                                                            y,
                                                                            test_size=0.2,
                                                                            random_state=42)

# # print(X_multi_train.shape)
# # print(X_multi_test.shape)
# # print(y_multi_train.shape)
# # print(y_multi_test.shape)

# Initialize and train the model
multi_model = LinearRegression()
multi_model.fit(X_multi_train,y_multi_train)

y_multi_pred = multi_model.predict(X_multi_test)
# print(y_multi_pred)

# Testing the performance of the model
multi_r2 = metrics.r2_score(y_multi_test,y_multi_pred)
multi_MSE = mean_squared_error(y_multi_test,y_multi_pred)

print(f'MultiLinear Regression R2 is {multi_r2}')
print(f'MultiLinear Regression MSE is {multi_MSE}')
print(f'MultiLinear Regression Coefficient is {multi_model.coef_}')
print(f'MultiLinear Regression Intercept is {multi_model.intercept_}')

""" Interpretation:
The R² value for the multiple linear regression model is approximately 0.68, which is significantly higher than the R² value for the simple linear regression model. 
This suggests that around 68% of the variance in house prices is explained by the selected features in the multiple linear regression model. 
The higher R² indicates that the multiple linear regression model captures more of the variability in house prices compared to the simple linear regression model.
The MSE for the multiple linear regression model is approximately 23.97 billion, which is substantially lower than the MSE for the simple linear regression model. 
The lower MSE indicates that the multiple linear regression model has a smaller average squared difference between the predicted house prices and the actual prices compared to the simple linear regression model. 
This suggests that the multiple linear regression model provides more accurate predictions.
Overall, the multiple linear regression model with the selected features demonstrates improved performance in terms of explained variance and prediction accuracy compared to the simple linear regression model. 
The coefficients can further provide insights into the relationships between the features and house prices, aiding in the interpretation of the model.
"""
# Polynominal Regression 
# # Select features based on correlation
x_poly  = data.drop(['id','date','price'], axis=1)
y_poly = data['price']

model = LinearRegression()
poly = PolynomialFeatures(degree=2)

x_poly_train,x_poly_test,y_poly_train,y_poly_test = train_test_split(x_poly,
                                                                     y_poly,
                                                                     test_size=0.35,
                                                                     random_state=40)
# Apply the fit_transform command to our input variable
x_ = poly.fit_transform(x_poly_train)
# print('x before transformation: ', x_poly_train)
# print('x after transformation: ', x_)

# Fiting polynominal regression to the data set
model.fit(x_,y_poly_train)
x_test_ = poly.fit_transform(x_poly_test)

y_predict_ = model.predict(x_test_)

# Testing for accuracy
poly_MSE = mean_squared_error(y_poly_test,y_predict_)
poly_r2 = metrics.r2_score(y_poly_test,y_predict_)

print(f'Polynominal RegressionMSE is {poly_MSE}')
print(f'Polynominal Regression R2 is {poly_r2}')
print(f'Polynominal Regression Coefficient is {model.coef_}')
print(f'Polynominal Intercept is {model.intercept_}')

""" Interpretation:
The R² value for the polynomial regression model is approximately 0.76, which is higher than both simple and multiple linear regression models. 
This suggests that around 76% of the variance in house prices is explained by the selected features in the polynomial regression model. 
The higher R² indicates that the polynomial regression model captures more of the variability in house prices compared to both linear regression models.
The MSE for the polynomial regression model is approximately 16.62 billion, which is lower than both linear regression models. 
The lower MSE indicates that the polynomial regression model has a smaller average squared difference between the predicted house prices and the actual prices compared to linear regression models. 
This suggests that the polynomial regression model provides more accurate predictions.
The polynomial regression model with the selected features demonstrates the best performance in terms of explained variance and prediction accuracy among the three models. The inclusion of higher-order polynomial features allows the model to capture more complex relationships between features and house prices, resulting in improved predictive power. 
However, it's important to note that polynomial regression models can be prone to overfitting, especially with high degrees of polynomial features, so caution should be exercised in selecting the appropriate degree for the polynomial features.
"""