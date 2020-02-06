import numpy  as np
import pandas as pd
import statsmodels.api as sm
from sklearn import metrics
# Import data into a DataFrame.
from sklearn.model_selection import train_test_split

path = "C:/Users/aiman/PycharmProjects/3948_A01052971_Predictive_Modelling" \
       "/Assignments/Assignment1/hw1_housingV2.csv"

df = pd.read_csv(path, skiprows=1,
                 encoding="ISO-8859-1", sep=',',
                 names=('SubClass', 'Zoning', 'LotFrontage', 'LotArea',
                        'Street', 'Alley', 'LotShape', 'LandContour',
                        'Utilities', 'LotConfig', 'LandSlope',
                        'Neighborhood', 'Condition1', 'Condition2',
                        'BldgType', 'HouseStyle', 'OverallQual',
                        'OverallCond', 'YearBuilt', 'YearRemodel',
                        'RoofStyle', 'RoofMat', 'Exterior1', 'Exterior2',
                        'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
                        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                        'BsmtFinType1', 'BsmtFinType2', 'Heating',
                        'HeatingQC', 'CentralAir', 'Electrical',
                        'FirstFlrSF', 'SecondFlrSF', 'LowQualFinSF',
                        'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
                        'FullBath', 'HalfBath', 'BedroomAbvGr',
                        'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
                        'Functional', 'Fireplaces', 'FireplaceQu',
                        'GarageType', 'GarageYrBlt', 'GarageFinish',
                        'GarageCars', 'GarageQual', 'GarageCond',
                        'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
                        'EnclosedPorch', 'ThreeSsnPorch', 'ScreenPorch',
                        'PoolArea', 'PoolQC', 'Fence', 'MiscFeature',
                        'MiscVal', 'MoSold', 'YrSold', 'SaleType',
                        'SaleCondition', 'SalePrice',
                        ))
# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head().transpose())  # View a snapshot of the data.
print(df.describe().transpose())  # View stats including counts which highlight
print(df.describe())


def convertNAcellsToNum(colName, df, measureType):
    """
    Impute invalid data cells with either the mode or mean or median values.
    :param colName:  the column (target) to inspect
    :param df: the dataframe
    :param measureType: value to impute
    :return:
    """
    # Create two new column names based on original column name.
    indicatorColName = 'm_' + colName  # Tracks whether imputed.
    imputedColName = 'imp_' + colName  # Stores original & imputed data.

    # Get mean or median depending on preference.
    imputedValue = 0
    if (measureType == "median"):
        imputedValue = df[colName].median()
    elif measureType == "mode":
        imputedValue = float(df[colName].mode())
    else:
        imputedValue = df[colName].mean()

    # Populate new columns with data.
    imputedColumn = []
    indictorColumn = []
    for i in range(len(df)):
        isImputed = False

        # mi_OriginalName column stores imputed & original data.
        if (np.isnan(df.loc[i][colName])):
            isImputed = True
            imputedColumn.append(imputedValue)
        else:
            imputedColumn.append(df.loc[i][colName])

        # mi_OriginalName column tracks if is imputed (1) or not (0).
        if (isImputed):
            indictorColumn.append(1)
        else:
            indictorColumn.append(0)

    # Append new columns to data frame but always keep original column.
    df[indicatorColName] = indictorColumn
    df[imputedColName] = imputedColumn
    return df


# df = convertNAcellsToNum('OverallQual', df, "mode")
# df = convertNAcellsToNum('MomEduc', df, "mean")
df = convertNAcellsToNum('OverallQual', df, "median")
print(df.head(10))
# ---------------------------------------------------------------------
# Column Targets for Analysis
# ---------------------------------------------------------------------
# You can include both the indicator 'm' and imputed 'imp' columns in your model.
# Sometimes both columns boost regression performance and sometimes they do not.

# X = df[['m_OverallQual', 'imp_OverallQual']].values

X = df[['FirstFlrSF',
        'SecondFlrSF',
        'FullBath',
        'TotRmsAbvGrd',
        ]].values

# Adding an intercept *** This is requried ***. Don't forget this step.
X = sm.add_constant(X)
y = df['SalePrice'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

# ---------------------------------------------------------------------
# end
# ---------------------------------------------------------------------

# Build and evaluate model.
model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)  # make the predictions by the model
print(model.summary())
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# print("\nbinned_statistic for median : \n", stats.binned_statistic(df['DadAge'].values,   bins = 4))
# df['dadAgeBin'] = pd.cut(x=df['DadAge'], bins=[17, 27, 37, 48])
# df['momAgeBin'] = pd.cut(x=df['MomAge'], bins=[13, 23, 33, 42])

# tempDf = df[['dadAgeBin', 'momAgeBin', 'sex']]  # Isolate columns
# Get dummies
# dummyDf = pd.get_dummies(tempDf, columns=['dadAgeBin', 'momAgeBin', 'sex'])
# df = pd.concat(([df, dummyDf]), axis=1)  # Join dummy df with original
# print(df)
