import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api       as sm
import matplotlib.pyplot     as plt
from scipy import stats
import numpy                 as np

# Import data into a DataFrame.
path = "../dataset/babysamp-98.txt"
df = pd.read_csv(path, skiprows=1,
                   sep='\t',
                   names=('MomAge', 'DadAge', 'MomEduc', 'MomMarital', 'numlive',
                          "dobmm", 'gestation', 'sex', 'weight', 'prenatalstart',
                          'orig.id', 'preemie'))
# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head())

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

THRESHOLD_Z =2.33
priceOutlierRows = viewAndGetOutliers(df, 'weight', THRESHOLD_Z, plt)
