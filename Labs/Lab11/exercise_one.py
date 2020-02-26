from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot       as plt
import pandas                  as pd
import numpy                   as np
from sklearn import metrics

PATH = "../dataset/"
CSV_DATA = "bank-additional-full.csv"
df = pd.read_csv(PATH + CSV_DATA,
                 skiprows=1,  # Don't include header row as part of data.
                 encoding="ISO-8859-1", sep=';',
                 names=(
"age", "job", "marital", "education", "default", "housing", "loan", "contact",
"month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome",
"emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"))
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(df.head())
print(df.describe().transpose())
print(df.info())

targetList = []
for i in range(0, len(df)):
    if (df.loc[i]['y'] == 'yes'):
        targetList.append(1)
    else:
        targetList.append(0)
df['target'] = targetList

tempDf = df[["job", "marital", "education", "default","housing", "loan", "contact", "month", "day_of_week", "poutcome"]]  # Isolate columns
dummyDf = pd.get_dummies(tempDf, columns=["job", "marital", "education", "default",
"housing", "loan", "contact", "month", "day_of_week", "poutcome"])  # Get dummies
df = pd.concat(([df, dummyDf]), axis=1)  # Join dummy df with original df

X = df[["duration",
        "campaign",
        "pdays",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "job_blue-collar",
        "default_no",
        "default_yes",
        "contact_cellular",
        "contact_telephone",
        "month_mar",
        "month_may",
        "month_nov",
        "day_of_week_mon",
        "poutcome_failure",
        "poutcome_success"]]
y = df[['target']]

XScaled   = MinMaxScaler().fit_transform(X)

# Split data.
X_train, X_test, y_train, y_test = train_test_split(
    XScaled, y, test_size=0.25, random_state=0)

# Perform logistic regression.
logisticModel = LogisticRegression(fit_intercept=True, random_state=0,
                                   solver='liblinear')

# Fit the model.
logisticModel.fit(X_train, y_train.values.ravel())

# Show model coefficients and intercept.
print("\nModel Intercept: ")
print(logisticModel.intercept_)

print("\nModel Coefficients: ")
print(logisticModel.coef_)

y_pred = logisticModel.predict(X_test)
y_prob = logisticModel.predict_proba(X_test)

# Show confusion matrix and accuracy scores.
cm = pd.crosstab(np.ravel(y_test), y_pred,
                 rownames=['Actual'],
                 colnames=['Predicted'])

print('\nAccuracy: ', metrics.accuracy_score(y_test, y_pred))
print("\nConfusion Matrix")
print(cm)

from sklearn.metrics import classification_report
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
