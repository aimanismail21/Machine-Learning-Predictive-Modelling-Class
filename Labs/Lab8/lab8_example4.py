import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api       as sm
import matplotlib.pyplot     as plt
from   scipy                 import stats
import numpy                 as np

PATH = "../dataset/"
CSV_DATA = "USA_Housing.csv"
dataset = pd.read_csv(PATH + CSV_DATA,
                      skiprows=1,  # Don't include header row as part of data.
                      encoding="ISO-8859-1", sep=',',
                      names=('Avg. Area Income', 'Avg. Area House Age',
                             'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', "Area Population",
                             'Price', "Address"))
# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(dataset.head())

# ------------------------------------------------------------------
# Show statistics, boxplot, extreme values and returns DataFrame
# row indexes where outliers exist.
# ------------------------------------------------------------------
def viewAndGetOutliers(df, colName, threshold, plt):
    # Show basic statistics.
    dfSub = df[[colName]]
    print("*** Statistics for " + colName)
    print(dfSub.describe())

    # Show boxplot.
    dfSub.boxplot(column=[colName])
    plt.title(colName)
    plt.show()

    # Note this is absolute 'abs' so it gets both high and low values.
    z = np.abs(stats.zscore(dfSub))
    rowColumnArray = np.where(z > threshold)
    rowIndices     = rowColumnArray[0]

    # Show outlier rows.
    print("\nOutlier row indexes for " + colName + ":")
    print(rowIndices)
    print("")

    # Show filtered and sorted DataFrame with outliers.
    dfSub = df.iloc[rowIndices]
    dfSorted = dfSub.sort_values([colName], ascending=[True])
    print("\nDataFrame rows containing outliers for " + colName + ":")
    print(dfSorted)
    return rowIndices

THRESHOLD_Z      = 3
priceOutlierRows = viewAndGetOutliers(dataset, 'Avg. Area Income',
                                      THRESHOLD_Z, plt)
