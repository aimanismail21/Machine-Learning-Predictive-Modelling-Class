import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import numpy as np

PATH = "../dataset/"
CSV_DATA = "computerPurchase.csv"
df = pd.read_csv(PATH + CSV_DATA,
                 skiprows=1,  # Don't include header row as part of data.
                 encoding="ISO-8859-1", sep=',',
                 names=("User ID", "Gender", "Age", "EstimatedSalary",
                        "Purchased"))



# Separate into x and y values.
X = df[["Age", "EstimatedSalary"]]
y = df['Purchased']

sc_x    = MinMaxScaler()
X_Scale = sc_x.fit_transform(X)
# Split data.
X_train, X_test, y_train, y_test = train_test_split(
    X_Scale, y, test_size=0.25, random_state=0)

# Perform logistic regression.
logisticModel = LogisticRegression(fit_intercept=True, random_state=0,
                                   solver='liblinear')

# Fit the model.
logisticModel.fit(X_train, y_train)
y_pred = logisticModel.predict(X_test)

# Show confusion matrix and accuracy scores.
cm = pd.crosstab(y_test, y_pred,
                               rownames=['Actual'],
                               colnames=['Predicted'])

print('\nAccuracy: ', metrics.accuracy_score(y_test, y_pred))
print("\nConfusion Matrix")
print(cm)

df2 = pd.DataFrame()


# def calculateProbabilities(X_test):
#     X1 = X_test[0]
#     X2 = X_test[1]
#     X3 = X_test[2]
#     X4 = X_test[3]
#
#     logit1 = -0.42172234 + -1.02163024 * X1 + 1.0430488 * X2 \
#              - 1.77994737 * X3 - 1.65927236 * X4
#     logit2 = 1.76983851 + 0.5420308 * X1 - 0.3599358 * X2 - 0.26681264 * X3 \
#              - 0.71830735 * X4
#     logit3 = -1.34811617 + 0.47959944 * X1 - 0.683113 * X2 + 2.04676001 * X3 \
#              + 2.37757971 * X4
#
#     denominator = 1 + np.exp(logit1) + np.exp(logit2) + np.exp(logit3)
#     prediction1 = np.exp(logit1) / denominator
#     prediction2 = np.exp(logit2) / denominator
#     prediction3 = np.exp(logit3) / denominator
#
#     print(str(prediction1) + "  " + str(prediction2) + "  " + str(prediction3))
#
#
# calculateProbabilities(Xtest_scaled[0, :])
# calculateProbabilities(Xtest_scaled[1, :])
# calculateProbabilities(Xtest_scaled[2, :])
# calculateProbabilities(Xtest_scaled[3, :])
