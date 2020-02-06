import math

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols


def performSimpleRegression():
    # Initialize collection of X & Y pairs like those used in example 5.
    data = [[0.2, 0.1], [0.32, 0.15], [0.38, 0.4], [0.41, 0.6], [0.43, 0.44]]

    # Create data frame.
    dfSample = pd.DataFrame(data, columns=['X', 'target'])

    # Create training set with 60% of data and test set with 40% of data.
    X_train, X_test, y_train, y_test = train_test_split(
        dfSample['X'], dfSample['target'], train_size=0.8
    )

    # Create DataFrame with test data.
    dataTrain = {"X": X_train, "target": y_train}
    dfTrain = pd.DataFrame(dataTrain, columns=['X', 'target'])

    # Generate model to predict target using X.
    model = ols('target ~ X', data=dfTrain).fit()
    y_prediction = model.predict(X_test)

    # Present X_test, y_test, y_predict and error sum of squares.
    data = {"X_test": X_test, "y_test": y_test, "y_prediction": y_prediction}
    dfResult = pd.DataFrame(data, columns=['X_test', 'y_test', 'y_prediction'])
    dfResult['x*y'] = (dfResult['X_test'] * dfResult['y_test'])
    dfResult['x^2'] = dfResult['X_test']**2
    dfResult['y_test - y_pred'] = (dfResult['y_test'] - dfResult['y_prediction'])
    dfResult['(y_test - y_pred)^2'] = (dfResult['y_test'] - dfResult['y_prediction']) ** 2
    # Present X_test, y_test, y_predict and error sum of squares.
    print(dfResult)

    # Manually calculate the deviation between actual and predicted values.
    sum_x = dfResult['X_test'].sum()
    sum_y = dfResult['y_test'].sum()
    sum_xy = dfResult['x*y'].sum()
    print(f"Summation X {sum_x}, Summation Y {sum_y}, Summation of XY{sum_xy}")
    sum_xsquare = dfResult['x^2'].sum()
    print(f"X Squared {sum_xsquare}")


    rmse = math.sqrt(dfResult['(y_test - y_pred)^2'].sum() / len(dfResult))
    print("RMSE is average deviation between actual and predicted values: "
          + str(rmse))

    # Show faster way to calculate deviation between actual and predicted values.
    rmse2 = math.sqrt(metrics.mean_squared_error(y_test, y_prediction))
    print("The automated root mean square error calculation is: " + str(rmse2))


performSimpleRegression()
