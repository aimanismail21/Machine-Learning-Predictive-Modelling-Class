import pandas               as pd
import numpy                as np
from sklearn                import model_selection
from sklearn.decomposition  import PCA
from sklearn.linear_model   import LinearRegression
from sklearn.metrics        import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing  import StandardScaler
PATH     = "../dataset/"
CSV_DATA = "USA_HousingMysterySet.csv"

# Drop null values.
df       = pd.read_csv(PATH + CSV_DATA).dropna()
df.info()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

x  = df[['Avg. Area Income', 'Avg. Area House Age',
       'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
       'Area Population']]
y = df.Salary

print("\nSalary stats: ")
print(y.describe())

# Drop the column with the independent variable (Salary),
# and columns for which we created dummy variables.
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')

# Define the feature set X.
X         = pd.concat([X_, x[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)

# Standardize the data.
X_scaled  = StandardScaler().fit_transform(X)

# Split into training and test sets
X_train, X_test , y_train, y_test = model_selection.train_test_split(X_scaled, y,
                                            test_size=0.25, random_state=1)

# Transform the data using PCA for first 80% of variance.
pca = PCA(.8)

X_reduced_train = pca.fit_transform(X_train)
# Show scaled X_test.
for i in range(0, X_test.shape[0]):
    print("\nScaled X_test: " + str(i))
    if i>3:
        break

    for j in range(0, X_test.shape[1]):
        print(X_test[i][j])

X_reduced_test  = pca.transform(X_test)

# Show X_test after pca reduction.
print("\nX_test_transformed")
print(X_reduced_test.shape)

print("\nPrincipal Components")
print(pca.components_)

print("\nExplained variance: ")
print(pca.explained_variance_)

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

from sklearn.metrics import r2_score
print("\nr2_score",r2_score(y_test,pred))

# For each principal component, calculate the VIF and save in dataframe
vif = pd.DataFrame()

# Show the VIF score for the principal components.
print()
vif["VIF Factor"] = [variance_inflation_factor(X_reduced_train, i) for i in range(X_reduced_train.shape[1])]
print(vif)
