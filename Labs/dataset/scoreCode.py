import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LinearRegression
from sklearn                 import metrics
import statsmodels.api       as sm
import numpy                 as np

PATH     = "/Users/pm/Desktop/DayDocs/2019_2020/PythonForDataAnalytics/workingData/"
CSV_DATA = "hw1_housing_mysteryfileV2.csv"
df       = pd.read_csv(PATH + CSV_DATA,
                      skiprows=1,       # Don't include header row as part of data.
                      encoding = "ISO-8859-1", sep=',',
                      names=(
                              
  "SubClass","Zoning","LotFrontage","LotArea","Street","Alley","LotShape",
  "LandContour","Utilities","LotConfig","LandSlope","Neighborhood",
  "Condition1","Condition2","BldgType","HouseStyle",
  "OverallQual","OverallCond","YearBuilt","YearRemodel","RoofStyle",
  "RoofMat","Exterior1","Exterior2","MasVnrType","MasVnrArea","ExterQual",
  "ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure",
  "BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir",
  "Electrical","FirstFlrSF","SecondFlrSF","LowQualFinSF","GrLivArea",
  "BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr",
  "KitchenAbvGr","KitchenQual","TotRmsAbvGrd","Functional","Fireplaces",
  "FireplaceQu","GarageType","GarageYrBlt","GarageFinish","GarageCars",
  "GarageQual","GarageCond","PavedDrive","WoodDeckSF","OpenPorchSF",
  "EnclosedPorch","ThreeSsnPorch","ScreenPorch","PoolArea","PoolQC",
  "Fence","MiscFeature","MiscVal","MoSold","YrSold","SaleType",
  "SaleCondition" ))


#------------------------------------------------------------
# Imputing
#------------------------------------------------------------
#MasVnrArea     = pd.to_numeric(df['MasVnrArea'])
imp_MasVnrArea  = []
m_MasVnrArea    = []

for i in range(0, len(df)):
    # This line is throwing errors:
    # or not df.loc[i]['MasVnrArea'].isdigit()
    if(pd.isnull(df.loc[i]['MasVnrArea']) ):
        imp_MasVnrArea.append(0) # subbing in 0 for now.
        m_MasVnrArea.append(1)   # indicated that imputed.
    # No need to impute.
    else:
        imp_MasVnrArea.append(pd.to_numeric(df.loc[i]['MasVnrArea'])) # add actual value.
        m_MasVnrArea.append(0)                         # No imputation.        

df['imp_MasVnrArea'] = imp_MasVnrArea
df['m_MasVnrArea']   = m_MasVnrArea

# Perform modelling for all rows.
target = -2.185e+06 + 1116.5745*df['YearRemodel']
+ 124.9106 *df["FirstFlrSF"]
+ 69.9303  *df["SecondFlrSF"]
+ 8189.4706*df["HalfBath"]
-5732.4052 *df["TotRmsAbvGrd"]
+ 1.571e+04 *df["Fireplaces"]
+ 75.9527*df["imp_MasVnrArea"]

# Make predicticted value column
df['PredictedSalePrice'] = target   

# Get single column and write to csv
df[['PredictedSalePrice']].to_csv(PATH + "predictions_Pat.csv", header=1, index=False)

