import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api       as sm
import matplotlib.pyplot     as plt
from scipy import stats
import numpy                 as np

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
# row indexes where outliers exist outside an upper and lower percentile.
# ------------------------------------------------------------------
def viewAndGetOutliersByPercentile(df, colName, lowerP, upperP, plt):
    # Show basic statistics.
    dfSub = df[[colName]]
    print("*** Statistics for " + colName)
    print(dfSub.describe())

    # Show boxplot.
    dfSub.boxplot(column=[colName])
    plt.title(colName)
    plt.show()

    # Get upper and lower perctiles and filter with them.
    up = df[colName].quantile(upperP)
    lp = df[colName].quantile(lowerP)
    outlierDf = df[(df[colName] < lp) | (df[colName] > up)]

    # Show filtered and sorted DataFrame with outliers.
    dfSorted = outlierDf.sort_values([colName], ascending=[True])
    print("\nDataFrame rows containing outliers for " + colName + ":")
    print(dfSorted)

    return lp, up  # return lower and upper percentiles

LOWER_PERCENTILE = 0.02
UPPER_PERCENTILE = 0.98
lp, up = viewAndGetOutliersByPercentile(df, 'gestation',
                                        LOWER_PERCENTILE, UPPER_PERCENTILE, plt)
