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


def setup_regression():
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
                       names=('SubClass', 'Zoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour',
                              'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
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

    # print(df.describe().round(4))

    # Binning
    # print(df['YearRemodel'].min())
    # print(df['YearRemodel'].max())
    # print(df['YearRemodel'].std())
    df['YearRemodel_bin'] = pd.cut(x=df['YearRemodel'], bins=[1950, 1960, 1970, 1980, 1990, 2000, 2010])
    temp_df = df['YearRemodel_bin']
    dummy_temp_df = pd.get_dummies(temp_df, columns=['YearRemodel_bin'])
    df = pd.concat((df, dummy_temp_df), axis=1)
    print(df)

    # Impute data
    # todo Can probably refine by checking outliers in the three categories, if outliers exist, using mean would not be ideal.
    df = convertNAcellsToNum('OverallQual', df, "mean")  # tried mode, mean
    df = convertNAcellsToNum('YearBuilt', df, "mean")
    df = convertNAcellsToNum("GrLivArea", df, "mean")
    # OQ, MEAN : YB, MEAN : GrLivArea: Mode RMSE 35782.99734322801 , R2     0.760

    # Dummy Variables
    columns_to_dummy = ['Zoning', 'LotShape', 'Neighborhood', 'HouseStyle', 'ExterQual', 'Foundation', 'HeatingQC', 'KitchenQual', 'GarageType', 'PavedDrive', 'SaleType']
    tempDf = df[columns_to_dummy]
    df = pd.get_dummies(df, columns=columns_to_dummy)

    # dummies to ignore: Utilities, LotConfig, Condition1, Condition2, BldgType, SaleCondition
    # -----------------------------------------------
    # # Compute the correlation matrix
    corr = df.corr()
    # Heatmap
    plt.subplots(figsize=(20,15))  # Expand the size of heatmap for better clarity
    sns.heatmap(corr)
    # sns.heatmap(corr,
    #             xticklabels=corr.columns,
    #             yticklabels=corr.columns)
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
def run_regression(X_values, Y_value, df):
    # Training Set - train_size 80/20 (as requested by Pat)
    X_training_set, X_test, Y_training_set, Y_test = train_test_split(X_values, Y_value, train_size=0.80, random_state=0)

    # Format the labeling and title of the graph
    plt.legend()
    plt.xlabel("Variables Used:")
    plt.title("Linear Regression Model")

    # OLS MODEL
    # data_training_set = {"LotArea": X_training_set, "SalePrice": Y_training_set}
    model = sm.OLS(Y_training_set, X_training_set).fit()
    y_prediction = model.predict(X_test)
    return model, y_prediction, Y_test

def convertNAcellsToNum(colName, df, measureType):
    # Create two new column names based on original column name.
    indicatorColName = 'm_'   + colName # Tracks whether imputed.
    imputedColName   = 'imp_' + colName # Stores original & imputed data.

    # Get mean or median depending on preference.
    imputedValue = 0
    if(measureType=="median"):
        imputedValue = df[colName].median()
    elif(measureType=="mode"):
        imputedValue = float(df[colName].mode())
    else:
        imputedValue = df[colName].mean()

    # Populate new columns with data.
    imputedColumn  = []
    indictorColumn = []
    for i in range(len(df)):
        isImputed = False

        # mi_OriginalName column stores imputed & original data.
        if(np.isnan(df.loc[i][colName])):
            isImputed = True
            imputedColumn.append(imputedValue)
        else:
            imputedColumn.append(df.loc[i][colName])

        # mi_OriginalName column tracks if is imputed (1) or not (0).
        if(isImputed):
            indictorColumn.append(1)
        else:
            indictorColumn.append(0)

    # Append new columns to dataframe but always keep original column.
    df[indicatorColName] = indictorColumn
    df[imputedColName]   = imputedColumn
    return df

def main():
    X_values, Y_value, df = setup_regression()
    model, y_prediction, Y_test = run_regression(X_values=X_values, Y_value=Y_value, df=df)
    print(model.summary())
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_prediction)))
    print(df.head().transpose())
    print(df.describe().transpose())


    # todo Code to export results to CSV
    PATH = "Assignment1/"
    OUT_FILE = "predictions_aiman.csv"  # Do not change.
    OUT_PATH = PATH + OUT_FILE  # Do not change.
    df.to_csv(r''+ OUT_PATH)


if __name__ == '__main__':
    main()

# Present X, Y Test and Error Sum of Squares
# data = {"X_test": X_test, "Y_test": Y_test, "Y_predicted":y_prediction}
# df_result = pd.DataFrame(data, columns=['X_test', 'Y_test', 'Y_predicted'])

