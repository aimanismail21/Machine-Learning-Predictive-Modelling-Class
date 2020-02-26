import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
PATH = "../dataset/"
CSV_DATA = "computerPurchase.csv"
df = pd.read_csv(PATH + CSV_DATA,
                 skiprows=1,  # Don't include header row as part of data.
                 encoding="ISO-8859-1", sep=',',
                 names=("User ID", "Gender", "Age", "EstimatedSalary",
                        "Purchased"))
count = 0

from scipy import stats
X_TransformedSalary = stats.boxcox(df['EstimatedSalary'])
df['TransformedSalary'] = X_TransformedSalary[0]
X = df[["Age", "TransformedSalary"]]
y = df[["Purchased"]]

import numpy as np
# enumerate splits - returns train and test arrays of indexes.
# scikit-learn k-fold cross-validation
from sklearn.model_selection import KFold
# data sample

# prepare cross validation with three folds and 1 as a random seed.
kfold = KFold(3, True, 1)
for train, test in kfold.split(df):
    X_train = X.iloc[train,:] # Gets all rows with train indexes.
    y_train = y.iloc[train,:]
    X_test =  X.iloc[test,:]
    y_test =  y.iloc[test,:]

    # Perform logistic regression.
    logisticModel = LogisticRegression(fit_intercept=True, random_state=0,
                                       solver='liblinear')
    # Fit the model.
    logisticModel.fit(X_train, np.ravel(y_train))

    y_pred = logisticModel.predict(X_test)
    y_prob = logisticModel.predict_proba(X_test)

    # Show confusion matrix and accuracy scores.
    cm = pd.crosstab(np.ravel(y_test), y_pred,
                                   rownames=['Actual'],
                                   colnames=['Predicted'])
    count += 1
    print("K-fold: " + str(count))
    print('\nAccuracy: ',metrics.accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix")
    print(cm)

    from sklearn.metrics import classification_report, roc_auc_score, roc_curve

    print(classification_report(y_test, y_pred))

    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_test, y_pred)

    print('Average precision-recall score: {0:0.2f}'.format(
          average_precision))

    # calculate scores
    auc = roc_auc_score(y_test, y_prob[:, 1],)
    print('Logistic: ROC AUC=%.3f' % (auc))

    # calculate roc curves
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_prob[:, 1])
    plt.plot(lr_fpr, lr_tpr, marker='.', label='ROC')
    plt.plot([0,1], [0,1], '--', label='50/50 Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

    # The magic happens here
    import matplotlib.pyplot as plt
    import scikitplot as skplt
    skplt.metrics.plot_cumulative_gain(y_test, y_prob)
    #skplt.metrics.plot_lift_curve(y_test, y_prob)
    plt.show()
