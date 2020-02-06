import matplotlib.pyplot as plt
import numpy                 as np
import pandas as pd
import statsmodels.api       as sm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from statsmodels.graphics.gofplots import qqplot

PATH = "../dataset/"
CSV_DATA = "USA_Housing.csv"
dataset = pd.read_csv(PATH + CSV_DATA,
                      skiprows=1,  # Don't include header row as part of data.
                      encoding="ISO-8859-1", sep=',',
                      names=(
                          'Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                          'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address'))

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
# Removed x vars
pd.set_option('display.width', 1000)
print(dataset.head(3))
print(dataset.describe())
X = dataset[['Avg. Area Income',
             'Avg. Area House Age',
             'Avg. Area Number of Rooms',
             'Area Population']].values
# # ----------------------------------------------
# # Show counts.
# print("Counts")
# print(dataset['Price'].value_counts(ascending=True))
# Adding an intercept *** This is requried ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X = sm.add_constant(X)

Y = dataset['Price'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)  # make the predictions by the model

print(model.summary())

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# -----------------------------------------------
# # Compute the correlation matrix
# corr = dataset.corr()
# # plot the heatmap
# sns.heatmap(corr,
#             xticklabels=corr.columns,
#             yticklabels=corr.columns)
# plt.show()


# Validation Plots
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


BINS = 8
TITLE = "USA House Pricing"
drawValidationPlots(TITLE, BINS, y_test, predictions)


# QQ PLots
def plotQQ(plt, title, y_test, predictions):
    residuals = y_test - predictions
    plt.title("Quantile-Quantial Residuals - " + title)
    qqplot(residuals)


plotQQ(plt, TITLE, y_test, predictions)
plt.show()


def test_equation(avg_area_income, avg_area_house_age, avg_area_number_rooms, area_population):
    price = -2.647e+06 + 21.6681 * avg_area_income + 1.658e+05 * avg_area_house_age + 1.216e+05 * (
        avg_area_number_rooms) + 15.2785 * area_population
    return price
print(test_equation(60000,10,3, 100000)) = 2203736.0
print(test_equation(30000,30,2, 100000)) = 4748093.0
print(test_equation(10000,50,2, 100000)) = 7630731.0