import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
import numpy as np

PATH = "C:/Users/aiman/PycharmProjects/3948_A01052971_Predictive_Modelling" \
       "/Labs/dataset/USA_Housing.csv"
dataset = pd.read_csv(PATH,
                      skiprows=1,  # Don't include header row as part of data.
                      encoding="ISO-8859-1", sep=',',
                      names=(
                          'Avg. Area Income',
                          'Avg. Area House Age',
                          'Avg. Area Number of Rooms',
                          'Avg. Area Number of Bedrooms',
                          'Area Population',
                          'Price',
                          'Address'
                      ))
# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)
print(dataset.head())
print(dataset.describe())
X = dataset[
    ['Avg. Area Income',
     'Avg. Area House Age',
     'Avg. Area Number of Rooms',
     'Avg. Area Number of Bedrooms',
     'Area Population',
     'Address'
     ]].values

# Adding an intercept *** This is requried ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X = sm.add_constant(X)

y = dataset['Price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)  # make the predictions by the model

print(model.summary())

print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))
