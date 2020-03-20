import pandas as pd
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score, mean_squared_error
import pandas               as pd
import numpy                as np
from sklearn                import model_selection
from sklearn.decomposition  import PCA
from sklearn.linear_model   import LinearRegression
from sklearn.metrics        import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing  import StandardScaler
from sklearn import preprocessing

import numpy  as np
import pandas as pd

PATH     = "../dataset/"
CSV_DATA = "USA_Housing.csv"
df       = pd.read_csv(PATH + CSV_DATA,
                      skiprows=1,       # Don't include header row as part of data.
                      encoding = "ISO-8859-1", sep=',',
                      names=('Avg. Area Income','Avg. Area House Age',
'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
                             "Area Population", 'Price', "Address")).dropna()
# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
df2 = df._get_numeric_data()

X   = df2.copy()
X.drop(['Price'], inplace=True, axis=1)
y   = df2.copy()
y   = y[['Price']]

# Drop null values.
df       = pd.read_csv(PATH + CSV_DATA).dropna()
df.info()


# Calculate and show VIF Scores for original data.
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print("\nOriginal VIF Scores")
print(vif)

# Standardize the data.
X_scaled  = StandardScaler().fit_transform(X)

# Split into training and test sets
X_train, X_test , y_train, y_test = model_selection.train_test_split(X_scaled, y,
                                            test_size=0.25, random_state=1)

scaler =preprocessing.StandardScaler().fit(X_train)
print("Mean")


# Transform the data using PCA for first 80% of variance.
pca = PCA(.8)
X_reduced_train = pca.fit_transform(X_train)
X_reduced_test  = pca.transform(X_test)[:,:7]

print("\nPrincipal Components")
print(pca.components_)

print("\nExplained variance: ")
print(pca.explained_variance_)

# Train regression model on training data
model = LinearRegression()
model.fit(X_reduced_train[:,:7], y_train)

# Prediction with test data
pred = model.predict(X_reduced_test)
sum_pred = sum(pred)
pred_avg = sum_pred/len(pred)
print("\nAverage Housing Price of Predicated Housing Prices")
print(np.mean(pred))
print(f"\nStandard Deviation of Housing Prices: {np.std(pred)}")
print("Min House Price:")
print(pred.min())
print("Max House Price")
print(pred.max())

# Show stats about the regression.
mse = mean_squared_error(y_test, pred)
RMSE = np.sqrt(mse)
print("\nRMSE: " + str(RMSE))

print("\nModel Coefficients")
print(model.coef_)

print("\nModel Intercept")
print(model.intercept_)

from sklearn.metrics import r2_score
print("\nr2_score",r2_score(y_test,pred))

# For each principal component, calculate the VIF and save in dataframe
vif = pd.DataFrame()

# Show the VIF score for the principal components.
print()
vif["VIF Factor"] = [variance_inflation_factor(X_reduced_train, i) for i in range(X_reduced_train.shape[1])]
print(vif)

pca_components = np.array(pca.components_)
coefficients = np.array(model.coef_)
transformation = coefficients.dot(pca_components)
print(f"Transformation {transformation}")
