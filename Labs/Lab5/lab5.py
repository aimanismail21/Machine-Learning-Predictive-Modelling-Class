import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
PATH = "C:/Users/aiman/PycharmProjects/3948_A01052971_Predictive_Modelling" \
       "/Labs/dataset/USA_Housing.csv"
dataset = pd.read_csv(PATH,
                      skiprows=1,  # Don't include header row as part of data.
                      encoding="ISO-8859-1", sep=',',
                      names=(
                          'Avg. Area Income',
                          'Avg. Area House Age',
                          'Avg. Area Number of Rooms',
                          'Avg. Area Number of Bedrooms',
                          'Area Population',
                          'Price',
                          'Address'
                      ))

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)
print(dataset.head(3))
print(dataset.describe())
X = dataset[
    ['Avg. Area Income',
     'Avg. Area House Age',
     'Avg. Area Number of Rooms',
     'Avg. Area Number of Bedrooms',
     'Area Population',
     'Price',
     'Address'
     ]].values

# Compute the correlation matrix
corr = dataset.corr()

# plot the heatmap
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)

plt.show()
print("\nFrequencies of Avg. Number of Bedrooms.")
print(dataset['Avg. Area Number of Bedrooms'].value_counts(ascending=True))