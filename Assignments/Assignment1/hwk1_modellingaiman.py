import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
import statsmodels.api as sm
import numpy                 as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from statsmodels.graphics.gofplots import qqplot

"""
Pat, please change the path of files in the model_one, model_two, model_three and model_four functions.

You will need to change the path of files in the main for output as well.
"""

def model_one():
    PATH = "hw1_housingV2.csv"
    IN_FILE = "hw1_housingV2.csv"  # Do not change.
    FULL_PATH = PATH + IN_FILE  # Do not change.
    MYSTERY_FILE = "hw1_housing_mysteryfileV2.csv"  # Do not change.
    OUT_FILE = "predictions_yourFirstName.csv"  # Do not change.
    IN_PATH = PATH + MYSTERY_FILE  # Do not change.
    OUT_PATH = PATH + OUT_FILE  # Do not change.

    df = pd.read_table(IN_FILE,
                       skiprows=1,
                       encoding="ISO-8859-1",
                       sep=",",
                       names=(
                           'SubClass', 'Zoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour',
                           'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                           'BldgType',
                           'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodel', 'RoofStyle',
                           'RoofMat', 'Exterior1', 'Exterior2', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
                           'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                           'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'FirstFlrSF', 'SecondFlrSF',
                           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                           'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
                           'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageQual',
                           'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ThreeSsnPorch',
                           'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold',
                           'SaleType', 'SaleCondition', 'SalePrice',
                       )
                       )
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    df['YearRemodel_bin'] = pd.cut(x=df['YearRemodel'], bins=[1950, 1960, 1970, 1980, 1990, 2000, 2010])
    temp_df = df['YearRemodel_bin']
    dummy_temp_df = pd.get_dummies(temp_df, columns=['YearRemodel_bin'])
    df = pd.concat((df, dummy_temp_df), axis=1)

    # Impute data
    # todo Can probably refine by checking outliers in the three categories, if outliers exist, using mean would not be ideal.
    df = convertNAcellsToNum('OverallQual', df, "mean")  # tried mode, mean
    df = convertNAcellsToNum('YearBuilt', df, "mean")
    df = convertNAcellsToNum("GrLivArea", df, "mean")

    # Dummy Variables
    columns_to_dummy = ['Zoning', 'LotShape', 'Neighborhood', 'HouseStyle', 'ExterQual', 'Foundation', 'HeatingQC',
                        'KitchenQual', 'GarageType', 'PavedDrive', 'SaleType']

    df = pd.get_dummies(df, columns=columns_to_dummy)


    # -----------------------------------------------
    # # Compute the correlation matrix
    corr = df.corr()
    # Heatmap
    plt.subplots(figsize=(20, 15))  # Expand the size of heatmap for better clarity
    sns.heatmap(corr)

    plt.show()

    # Set the predictor values
    X_values = df[['LotArea',
                   'YearRemodel',
                   'FirstFlrSF',
                   'SecondFlrSF',
                   'FullBath',
                   'HalfBath',
                   'BedroomAbvGr',
                   'TotRmsAbvGrd',
                   'Fireplaces',
                   'GarageCars',
                   'WoodDeckSF',
                   'OpenPorchSF',
                   'm_OverallQual',
                   'imp_OverallQual',
                   'm_YearBuilt',
                   'imp_YearBuilt',
                   'm_GrLivArea',
                   'imp_GrLivArea',
                   'Neighborhood_NridgHt',
                   'Neighborhood_NoRidge',
                   'Neighborhood_OldTown',
                   'Neighborhood_NAmes',
                   'Neighborhood_Edwards',
                   'SaleType_New',
                   'SaleType_WD',
                   'PavedDrive_N',
                   'PavedDrive_Y',
                   'GarageType_Attchd',
                   'GarageType_Detchd',
                   'KitchenQual_Ex',
                   'KitchenQual_TA',
                   'HeatingQC_Ex',
                   'HeatingQC_TA',
                   'Foundation_PConc',
                   'ExterQual_Ex',
                   'ExterQual_Gd',
                   'ExterQual_TA',
                   'HouseStyle_2Story',
                   'LotShape_IR1',
                   'Zoning_RL'
                   ]].values
    X_values = sm.add_constant(X_values)
    Y_value = df['SalePrice'].values  # This is the target or predicted value Y

    return X_values, Y_value, df


def model_two():
    PATH = "hw1_housingV2.csv"
    IN_FILE = "hw1_housingV2.csv"  # Do not change.
    FULL_PATH = PATH + IN_FILE  # Do not change.
    MYSTERY_FILE = "hw1_housing_mysteryfileV2.csv"  # Do not change.
    OUT_FILE = "predictions_yourFirstName.csv"  # Do not change.
    IN_PATH = PATH + MYSTERY_FILE  # Do not change.
    OUT_PATH = PATH + OUT_FILE  # Do not change.

    df = pd.read_table(IN_FILE,
                       skiprows=1,
                       encoding="ISO-8859-1",
                       sep=",",
                       names=(
                           'SubClass', 'Zoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour',
                           'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                           'BldgType',
                           'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodel', 'RoofStyle',
                           'RoofMat', 'Exterior1', 'Exterior2', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
                           'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                           'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'FirstFlrSF', 'SecondFlrSF',
                           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                           'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
                           'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageQual',
                           'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ThreeSsnPorch',
                           'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold',
                           'SaleType', 'SaleCondition', 'SalePrice',
                       )
                       )
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)


    df['YearRemodel_bin'] = pd.cut(x=df['YearRemodel'],
                                   bins=[1950, 1955, 1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010])
    temp_df = df['YearRemodel_bin']
    dummy_temp_df = pd.get_dummies(temp_df, columns=['YearRemodel_bin'])
    df = pd.concat((df, dummy_temp_df), axis=1)


    df = convertNAcellsToNum('OverallQual', df, "mean")  # tried mode, mean
    df = convertNAcellsToNum('YearBuilt', df, "mean")
    df = convertNAcellsToNum("GrLivArea", df, "mean")

    columns_to_dummy = ['LotShape', 'Neighborhood', 'ExterQual', 'Foundation', 'HeatingQC',
                        'KitchenQual', 'GarageType', 'PavedDrive', 'SaleType']
    tempDf = df[columns_to_dummy]
    df = pd.get_dummies(df, columns=columns_to_dummy)

    corr = df.corr()
    # Heatmap
    plt.subplots(figsize=(20, 15))  # Expand the size of heatmap for better clarity
    sns.heatmap(corr)

    plt.show()


    X_values = df[['LotArea',
                   'YearRemodel',
                   'FirstFlrSF',
                   'SecondFlrSF',
                   'FullBath',
                   'BedroomAbvGr',
                   'Fireplaces',
                   'GarageCars',
                   'WoodDeckSF',
                   'Neighborhood_NridgHt',
                   'Neighborhood_NoRidge',
                   'Neighborhood_OldTown',
                   'Neighborhood_Edwards',
                   'SaleType_New',
                   'PavedDrive_N',
                   'GarageType_Attchd',
                   'KitchenQual_Ex',
                   'HeatingQC_Ex',
                   'Foundation_PConc',
                   'ExterQual_Ex',
                   'ExterQual_Gd',
                   'LotShape_IR1',
                   ]].values
    X_values = sm.add_constant(X_values)
    Y_value = df['SalePrice'].values  # This is the target or predicted value Y

    return X_values, Y_value, df


def model_three():
    """
    Remove very low p scores
    :return:
    """
    PATH = "hw1_housingV2.csv"
    IN_FILE = "hw1_housingV2.csv"  # Do not change.
    FULL_PATH = PATH + IN_FILE  # Do not change.
    MYSTERY_FILE = "hw1_housing_mysteryfileV2.csv"  # Do not change.
    OUT_FILE = "predictions_yourFirstName.csv"  # Do not change.
    IN_PATH = PATH + MYSTERY_FILE  # Do not change.
    OUT_PATH = PATH + OUT_FILE  # Do not change.

    df = pd.read_table(IN_FILE,
                       skiprows=1,
                       encoding="ISO-8859-1",
                       sep=",",
                       names=(
                           'SubClass', 'Zoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour',
                           'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                           'BldgType',
                           'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodel', 'RoofStyle',
                           'RoofMat', 'Exterior1', 'Exterior2', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
                           'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                           'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'FirstFlrSF', 'SecondFlrSF',
                           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                           'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
                           'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageQual',
                           'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ThreeSsnPorch',
                           'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold',
                           'SaleType', 'SaleCondition', 'SalePrice',
                       )
                       )
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    df['YearRemodel_bin'] = pd.cut(x=df['YearRemodel'],
                                   bins=[1950, 1955, 1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010])
    temp_df = df['YearRemodel_bin']
    dummy_temp_df = pd.get_dummies(temp_df, columns=['YearRemodel_bin'])
    df = pd.concat((df, dummy_temp_df), axis=1)

    # Impute data

    df = convertNAcellsToNum('OverallQual', df, "mean")  # tried mode, mean
    df = convertNAcellsToNum('YearBuilt', df, "mean")
    df = convertNAcellsToNum("GrLivArea", df, "mean")


    # Dummy Variables
    columns_to_dummy = ['Neighborhood', 'Foundation', 'ExterQual', 'PavedDrive', 'KitchenQual']
    tempDf = df[columns_to_dummy]
    df = pd.get_dummies(df, columns=columns_to_dummy)

    corr = df.corr()
    # Heatmap
    plt.subplots(figsize=(20, 15))  # Expand the size of heatmap for better clarity
    sns.heatmap(corr)

    plt.show()





    X_values = df[['LotArea',
                   'YearRemodel',
                   'FirstFlrSF',
                   'SecondFlrSF',
                   'Fireplaces',
                   'GarageCars',
                   'Neighborhood_NridgHt',
                   'Neighborhood_NoRidge',
                   'Neighborhood_OldTown',
                   'Neighborhood_Edwards',
                   'PavedDrive_N',
                   'KitchenQual_Ex',
                   'Foundation_PConc',
                   'ExterQual_Ex',
                   'ExterQual_TA'
                   ]].values
    X_values = sm.add_constant(X_values)
    Y_value = df['SalePrice'].values  # This is the target or predicted value Y

    return X_values, Y_value, df


def model_four():

    PATH = "hw1_housingV2.csv"
    IN_FILE = "hw1_housingV2.csv"  # Do not change.
    FULL_PATH = PATH + IN_FILE  # Do not change.
    MYSTERY_FILE = "hw1_housing_mysteryfileV2.csv"  # Do not change.
    OUT_FILE = "predictions_yourFirstName.csv"  # Do not change.
    IN_PATH = PATH + MYSTERY_FILE  # Do not change.
    OUT_PATH = PATH + OUT_FILE  # Do not change.

    df = pd.read_table(IN_FILE,
                       skiprows=1,
                       encoding="ISO-8859-1",
                       sep=",",
                       names=(
                           'SubClass', 'Zoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour',
                           'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                           'BldgType',
                           'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodel', 'RoofStyle',
                           'RoofMat', 'Exterior1', 'Exterior2', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
                           'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                           'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'FirstFlrSF', 'SecondFlrSF',
                           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                           'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
                           'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageQual',
                           'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ThreeSsnPorch',
                           'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold',
                           'SaleType', 'SaleCondition', 'SalePrice',
                       )
                       )
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Binning
    df['YearRemodel_bin'] = pd.cut(x=df['YearRemodel'],
                                   bins=[1950, 1960, 1970, 1980, 1990, 2000, 2010])
    temp_df = df['YearRemodel_bin']
    dummy_temp_df = pd.get_dummies(temp_df, columns=['YearRemodel_bin'])
    df = pd.concat((df, dummy_temp_df), axis=1)

    # Impute data
    df = convertNAcellsToNum('OverallQual', df, "median")
    df = convertNAcellsToNum('YearBuilt', df, "median")
    df = convertNAcellsToNum("GrLivArea", df, "mean")

    corr = df.corr()
    # Heatmap before Dummy Variables
    plt.subplots(figsize=(20, 15))  # Expand the size of heatmap for better clarity
    sns.heatmap(corr)

    # Dummy Variables
    columns_to_dummy = ['Neighborhood', 'ExterQual', 'KitchenQual']
    tempDf = df[columns_to_dummy]
    dummy_df = pd.get_dummies(tempDf, columns=columns_to_dummy)
    df = pd.concat(([df, dummy_df]), axis=1)

    corr = df.corr()
    # Heatmap
    plt.subplots(figsize=(20, 15))  # Expand the size of heatmap for better clarity
    sns.heatmap(corr)

    plt.show()

    X_values = df[['LotArea',
                   'FirstFlrSF',
                   'SecondFlrSF',
                   'Fireplaces',
                   'GarageCars',
                   'KitchenQual_Ex',
                   'ExterQual_Ex',
                   'ExterQual_TA',
                   'Neighborhood_NridgHt',
                   'Neighborhood_NoRidge'
                   ]].values
    X_values = sm.add_constant(X_values)
    Y_value = df['SalePrice'].values  # This is the target or predicted value Y

    return X_values, Y_value, df


def run_regression(X_values, Y_value, df):
    # Training Set - train_size 80/20 (as requested by Pat)
    X_training_set, X_test, Y_training_set, Y_test = train_test_split(X_values, Y_value, train_size=0.80,
                                                                      random_state=0)

    # Format the labeling and title of the graph
    plt.legend()
    plt.xlabel("Variables Used:")
    plt.title("Linear Regression Model")

    # OLS MODEL
    # data_training_set = {"LotArea": X_training_set, "SalePrice": Y_training_set}
    model = sm.OLS(Y_training_set, X_training_set).fit()
    y_prediction = model.predict(X_test)
    return model, y_prediction, Y_test, X_test


def convertNAcellsToNum(colName, df, measureType):
    # Create two new column names based on original column name.
    indicatorColName = 'm_' + colName  # Tracks whether imputed.
    imputedColName = 'imp_' + colName  # Stores original & imputed data.

    # Get mean or median depending on preference.
    imputedValue = 0
    if (measureType == "median"):
        imputedValue = df[colName].median()
    elif (measureType == "mode"):
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

    # Append new columns to dataframe but always keep original column.
    df[indicatorColName] = indictorColumn
    df[imputedColName] = imputedColumn
    return df


def plotPredictionVsActual(plt, title, y_test, predictions):
    plt.scatter(y_test, predictions)
    plt.legend()
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title('Predicted (Y) vs. Actual (X): ' + title)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')


def plotResidualsVsActual(plt, title, y_test, predictions):
    residuals = y_test - predictions
    plt.scatter(y_test, residuals, label='Residuals vs Actual')
    plt.xlabel("Actual")
    plt.ylabel("Residual")
    plt.title('Error Residuals (Y) vs. Actual (X): ' + title)
    plt.plot([y_test.min(), y_test.max()], [0, 0], 'k--')


def plotResidualHistogram(plt, title, y_test, predictions, bins):
    residuals = y_test - predictions
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.hist(residuals, label='Residuals vs Actual', bins=bins)
    plt.title('Error Residual Frequency: ' + title)
    plt.plot()


def drawValidationPlots(title, bins, y_test, predictions):
    # Define number of rows and columns for graph display.
    plt.subplots(nrows=1, ncols=3, figsize=(12, 5))

    plt.subplot(1, 3, 1)  # Specfy total rows, columns and image #
    plotPredictionVsActual(plt, title, y_test, predictions)

    plt.subplot(1, 3, 2)  # Specfy total rows, columns and image #
    plotResidualsVsActual(plt, title, y_test, predictions)

    plt.subplot(1, 3, 3)  # Specfy total rows, columns and image #
    plotResidualHistogram(plt, title, y_test, predictions, bins)
    plt.show()




def plotQQ(plt, title, y_test, predictions):
    residuals = y_test - predictions
    qqplot(residuals)


def main():
    #Model 1 Uncomment to USE
    # print("-----------------------------------------------------------------------\nModel 1")
    # X_values, Y_value, df = model_one()
    # model, y_prediction, Y_test, X_test = run_regression(X_values=X_values, Y_value=Y_value, df=df)
    # print(model.summary())
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_prediction)))

    #Model 2 Uncomment to USE
    # print("----------------------------------------------------------------\nModel 2")
    # X_values, Y_value, df = model_two()
    # model, y_prediction, Y_test, X_test = run_regression(X_values=X_values, Y_value=Y_value, df=df)
    # print(model.summary())
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_prediction)))

    # Model 3 Uncomment to USE
    print("----------------------------------------------------------------\nModel 3")
    X_values, Y_value, df = model_three()
    model, y_prediction, Y_test, X_test = run_regression(X_values=X_values, Y_value=Y_value, df=df)
    print(model.summary())
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_prediction)))

    # Model 4 Uncomment to USE
    # print("----------------------------------------------------------------\nModel 4")
    # X_values, Y_value, df = model_four()
    # model, y_prediction, Y_test, X_test = run_regression(X_values=X_values, Y_value=Y_value, df=df)
    # print(model.summary())
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_prediction)))

    # Draw Validation Plots
    BINS = 15
    TITLE = "Sale Price"
    drawValidationPlots(TITLE, BINS, Y_test, y_prediction)
    plotQQ(plt, TITLE, Y_test, y_prediction)
    plt.title("Quantile-Quantial Residuals - " + TITLE)
    plt.show()

    # Y-Predicated (SalePrice Prediction) Output
    sales_prediction = {'PredictedSalePrice': y_prediction}
    output_file = pd.DataFrame(sales_prediction, columns=['PredictedSalePrice'])
    output_file.describe().transpose()

    # todo create new df for csv output
    # todo Code to export results to CSV, need to just provide predicted results
    PATH = "hw1_housingV2.csv"
    IN_FILE = "hw1_housingV2.csv"  # Do not change.
    FULL_PATH = PATH + IN_FILE  # Do not change.
    MYSTERY_FILE = "hw1_housing_mysteryfileV2.csv"  # Do not change.
    OUT_FILE = "predictions_yourFirstName.csv"  # Do not change.
    IN_PATH = PATH + MYSTERY_FILE  # Do not change.
    OUT_PATH = PATH + OUT_FILE  # Do not change.
    output_file.to_csv(
        r'C:\Users\aiman\PycharmProjects\3948_A01052971_Predictive_Modelling\Assignments\Assignment1\predictions_aiman.csv', index=None)


if __name__ == '__main__':
    main()
