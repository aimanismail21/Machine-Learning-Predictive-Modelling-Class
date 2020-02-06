import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api       as sm
import matplotlib.pyplot     as plt
from   scipy                 import stats
import numpy                 as np

PATH = "/Users/pm/Desktop/DayDocs/2019_2020/PythonForDataAnalytics/workingData/"
CSV_DATA = "hw1_housing.csv"
dataset  = pd.read_csv(PATH + CSV_DATA,
                       skiprows=1,  # Don't include header row as part of data.
                       encoding="ISO-8859-1", sep=',',
                       names=( 
        "SubClass","Zoning","LotFrontage","LotArea","Street","Alley",
        "LotShape","LandContour","Utilities","LotConfig","LandSlope",
        "Neighborhood","Condition1","Condition2","BldgType","HouseStyle",
        "OverallQual","OverallCond","YearBuilt","YearRemodel","RoofStyle",
        "RoofMat","Exterior1","Exterior2","MasVnrType","MasVnrArea","ExterQual",
        "ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure",
        "BsmtFinType1","BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF",
        "TotalBsmtSF","Heating","HeatingQC","CentralAir","Electrical","FirstFlrSF",
        "SecondFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath",
        "FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","KitchenQual",
        "TotRmsAbvGrd","Functional","Fireplaces","FireplaceQu","GarageType",
        "GarageYrBlt","GarageFinish","GarageCars","GarageArea","GarageQual",
        "GarageCond","PavedDrive","WoodDeckSF","OpenPorchSF","EnclosedPorch",
        "ThreeSsnPorch","ScreenPorch","PoolArea","PoolQC","Fence","MiscFeature",
        "MiscVal","MoSold","YrSold","SaleType","SaleCondition","SalePrice"  ))

# Show all rows.
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
print(dataset.head().transpose())
print(dataset.describe().transpose())