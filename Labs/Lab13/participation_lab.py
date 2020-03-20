import pandas as pd
import numpy  as np
from sklearn.linear_model                 import LinearRegression
from sklearn import model_selection
from sklearn.decomposition                import PCA
from sklearn.metrics                      import mean_squared_error
from sklearn.preprocessing                import StandardScaler

PATH     = "../dataset/"
CSV_DATA = "USA_Housing.csv"
df       = pd.read_csv(PATH + CSV_DATA,
                      skiprows=1,       # Don't include header row as part of data.
                      encoding = "ISO-8859-1", sep=',',
                      names=('Avg. Area Income','Avg. Area House Age', 'Avg. Area Number of Rooms',
                             'Avg. Area Number of Bedrooms', "Area Population", 'Price', "Address"))
# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head())

X = df[['Avg. Area Income', 'Avg. Area House Age',
       'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
       'Area Population']]

y = df[['Price']]

# Standardize the data.
X_scaled  = StandardScaler().fit_transform(X)

# Split into training and test sets
X_train, X_test , y_train, y_test = model_selection.train_test_split(X_scaled, y,
                                            test_size=0.25, random_state=1)

# Transform the data using PCA for first 80% of variance.
pca             = PCA(n_components=4)
X_reduced_train = pca.fit_transform(X_train)
X_reduced_test  = pca.transform(X_test)
print(X_reduced_train.shape)
print(y_train.shape)

print("\nExplained variance")
print(pca.explained_variance_)

print("\nPrincipal Components")
print(pca.components_)

# Train regression model on training data
model = LinearRegression()
model.fit(X_reduced_train, y_train)

print("\nModel Coefficients")
print(model.coef_)

print("\nModel Intercept")
print(model.intercept_)

# Prediction with test data
pred = model.predict(X_reduced_test)
print("\nShow predictions")
print(pred)

# Show stats about the regression.
mse = mean_squared_error(y_test, pred)
RMSE = np.sqrt(mse)
print("\nRMSE: " + str(RMSE))

from sklearn.metrics import r2_score, mean_squared_error

print("\nr2_score",r2_score(y_test,pred))
