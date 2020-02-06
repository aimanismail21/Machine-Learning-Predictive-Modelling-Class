import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api       as sm
import matplotlib.pyplot     as plt
from scipy import stats
import numpy                 as np

PATH = "../dataset/"
CSV_DATA = "wnba.csv"
dataset = pd.read_csv(PATH + CSV_DATA,
                      skiprows=1,  # Don't include header row as part of data.
                      encoding="ISO-8859-1", sep=',',
                      names=("PLAYER", "GPS", "PTS"))
# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(dataset.head(30))


# ------------------------------------------------------------------
# Show statistics, boxplot, extreme values and returns DataFrame
# row indexes where outliers exist.
# ------------------------------------------------------------------
# def viewAndGetOutliers(df, colName, threshold, plt):
#     # Show basic statistics.
#     dfSub = df[['GPS', 'PTS']]
#     print("*** Statistics for " + colName)
#     print(dfSub.describe())
#
#     # Show boxplot.
#     dfSub.boxplot(column=[colName])
#     plt.title(colName)
#     plt.show()
#
#     # Note this is absolute 'abs' so it gets both high and low values.
#     z = np.abs(stats.zscore(dfSub))
#     rowColumnArray = np.where(z > threshold)
#     rowIndices = rowColumnArray[0]
#
#     # Show outlier rows.
#     print("\nOutlier row indexes for " + colName + ":")
#     print(rowIndices)
#     print("")
#
#     # Show filtered and sorted DataFrame with outliers.
#     dfSub = df.iloc[rowIndices]
#     dfSorted = dfSub.sort_values([colName], ascending=[True])
#     print("\nDataFrame rows containing outliers for " + colName + ":")
#     print(dfSorted)
#     return rowIndices


# Show revised results.

# THRESHOLD_Z = 2.33
# priceOutlierRows = viewAndGetOutliers(dataset[['PLAYER', 'GPS', 'PTS', 'GPS_ADJUSTED', 'PTS_ADJUSTED']],
#                                       'GPS', THRESHOLD_Z, plt)
#
#
# lp, up = viewAndGetOutliers(dataset
# [['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
#   'Avg. Area Number of Bedrooms', "Area Population", 'Price']], 'Price',
#    LOWER_PERCENTILE, UPPER_PERCENTILE, plt)
#
# print("Lower " + str(LOWER_PERCENTILE) + " percentile limit: " + str(lp))
# print("Upper " + str(UPPER_PERCENTILE) + " percentile limit: " + str(up))
# print("Dataframe length: " + str(len(dataset)))

df_filtered = dataset[(dataset["GPS"] <37) & (dataset["PTS"] < 861)]
print("\nDataframe length after filtering: " + str(len(df_filtered)))
print(df_filtered)
