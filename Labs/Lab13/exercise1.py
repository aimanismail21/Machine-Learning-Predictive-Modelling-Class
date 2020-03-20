import pandas as pd
import numpy  as np

# Can get from PyCharm terminal.
from factor_analyzer import FactorAnalyzer

# Read data.
from sklearn.linear_model import LinearRegression
PATH        = "../dataset/"
CSV_DATA    = "usCityData.csv"
data        = pd.read_csv(PATH + CSV_DATA, sep=',')

# Create data frame without ID and Satisfaction columns.
df = data.copy()
# Columns
# crim,zn,indus,nox,rm,age,dis,rad,tax,ptratio,lstat,medv

# Display all columns of the data frame.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head(2))

# Bartlett's test of sphericity tests the hypothesis that your correlation matrix
# is an identity matrix. If the correlation matrix is an identity matrix the
# columns are unrelated and are therefore unsuitable for structure detection.
# If the test is insignificant do not use factor analysis.
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value, p_value=calculate_bartlett_sphericity(df)

print("\nBartlett's test chi-square value: ")
print(chi_square_value)

print("\nBartlett's test p-value: ")
print(p_value)

# Kaiser-Meyer-Olkin (KMO) test measures the proportion of variance among
# variables that might be common variance. The lower the proportion,
# the more suited your data is to Factor Analysis. Factor analysis is suitable
# for scores of 0.6 (and sometimes 0.5) and above.
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(df)
print("\nKaiser-Meyer-Olkin (KMO) Test: Suitability of data for factor analysis.")
print(kmo_model)

# Create factor analysis and examine loading vectors and Eigenvalues.
fa = FactorAnalyzer(rotation=None)
fa.fit(df)
print("\nFactors:")
print(fa.loadings_)

ev, v = fa.get_eigenvalues()
print("\nEignenvalues:")
print(ev)

# Pick factors where eigenvalues are greater than 1.
fa = FactorAnalyzer(rotation="varimax", n_factors=3)
fa.fit(df)

# Create formatted factor loading matrix.
dfFactors = pd.DataFrame(fa.loadings_)
dfFactors['Categories'] = df.keys().values.tolist()
dfFactors = dfFactors.rename(columns={0:'Factor 1',
          1:'Factor 2', 2:'Factor 3'})
print("\nFactors: ")
print(dfFactors)

    # Display common variance.
print("\nVariance Explanation")
variances = fa.get_factor_variance()
varianceDf = pd.DataFrame(data=variances)
varianceDf = varianceDf.rename(columns={0:'F1',
          1:'F2', 2:'F3'})
varianceDf['Totals'] = ['Eigenvalues', '% variance', 'cumulative variance']
print(varianceDf)

# Display RMSE, R^2, model coefficients and intercept.
def showModelSummary(model, y_test, X_test_tranformed):
    print("\n****** MODEL SUMMARY ******")
    pred = model.predict(X_test_tranformed)

    # Show stats about the regression.
    from sklearn.metrics import mean_squared_error
    mse  = mean_squared_error(y_test, pred)
    RMSE = np.sqrt(mse)
    print("\nRMSE: " + str(RMSE))

    from sklearn.metrics import r2_score
    print("\nR^2:  ",r2_score(y_test,pred))

    print("\nModel Coefficients:")
    print(model.coef_)

    print("\nModel Intercept:")
    print(model.intercept_)

# Display p-values for model coefficients.
def showCoefficientPValues(y_train, X_train_transformed):
    import statsmodels.api       as sm
    X2       = sm.add_constant(X_train_transformed)
    model    = sm.OLS(y_train, X2)
    fii      = model.fit()
    p_values = fii.summary2().tables[1]['P>|t|']
    print("\nModel p-values: ")
    print(p_values)

# Split data before it is transformed.
from    sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(df, data['medv'],
                                               test_size=0.3, random_state=1)
# Transform data with factor components.
X_train_transformed = fa.fit_transform(X_train)
X_test_tranformed   = fa.transform(X_test)

#----------------------------------------------------------
# Build first model
#----------------------------------------------------------
# Train regression model on training data
model = LinearRegression()
model.fit(X_train_transformed, y_train)

# Show model statistics.
showModelSummary(model, y_test, X_test_tranformed)

# Check coefficient significance.
showCoefficientPValues(y_train, X_train_transformed)

#----------------------------------------------------------
# Build second model without insignificant variable.
#----------------------------------------------------------
# Builds labelled DataFrame with signficant latent variables from
# factor matrix.
def dropInsignificantX(X_transformed):
    # Builds DataFrame from matrix.
    dfX = pd.DataFrame(data=X_transformed)

    # Labels columns and drops insignificant column.
    dfX = dfX.rename(columns={0:'Purchase',
       1:'Marking', 2:'Post Purchase', 3:'Product Position'})

    del    dfX['Post Purchase']
    return dfX

# Prepare significant X values for regression.
trainDF = dropInsignificantX(X_train_transformed)
testDF  = dropInsignificantX(X_test_tranformed)

# Train regression model on training data
model = LinearRegression()
model.fit(trainDF, y_train)

# Show model statistics.
showModelSummary(model, y_test, testDF)

# Check coefficient significance.
showCoefficientPValues(y_train, trainDF.values)

print("FA Factor Variance")
print(fa.get_factor_variance())
print("FA Get Communalities")
print(fa.get_communalities())
print("FA Uniqunesses")
print(fa.get_uniquenesses())
