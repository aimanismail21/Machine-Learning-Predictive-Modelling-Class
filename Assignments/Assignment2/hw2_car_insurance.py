import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def main():
    model_one(1)


def model_one(number):
    print(f"\nExecuting Model {number}")
    PATH = "carInsurance.csv"
    df = pd.read_csv(PATH,
                     skiprows=1,
                     encoding="ISO-8859-1",
                     sep=',',
                     names=(
                         "ID", "KIDSDRIV", "BIRTH", "AGE", "HOMEKIDS", "YOJ",
                         "INCOME",
                         "PARENT1", "HOME_VAL", "MSTATUS", "GENDER",
                         "EDUCATION",
                         "OCCUPATION", "TRAVTIME", "CAR_USE", "BLUEBOOK",
                         "TIF",
                         "CAR_TYPE", "RED_CAR", "OLDCLAIM", "CLM_FREQ",
                         "REVOKED",
                         "MVR_PTS", "CLM_AMT", "CAR_AGE", "CLAIM_FLAG",
                         "URBANICITY"))
    # Show all columns.
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)

    # Exploratory Analysis
    def exploratory_analysis(df):
        print(df.columns)  # list all column names
        print(df.shape)  # get number of rows and columns
        print(df.info())  # additional info about dataframe
        print(
            df.describe())  # statistical description, only for numeric values
        columns = ["ID", "KIDSDRIV", "BIRTH", "AGE", "HOMEKIDS", "YOJ",
                   "INCOME",
                   "PARENT1", "HOME_VAL", "MSTATUS", "GENDER", "EDUCATION",
                   "OCCUPATION", "TRAVTIME", "CAR_USE", "BLUEBOOK", "TIF",
                   "CAR_TYPE", "RED_CAR", "OLDCLAIM", "CLM_FREQ", "REVOKED",
                   "MVR_PTS", "CLM_AMT", "CAR_AGE", "CLAIM_FLAG",
                   "URBANICITY"]
        for column in columns:
            print(
                df[column].value_counts(
                    dropna=False))  # count unique values in a

    # End of Exploratory Analysis

    # Convert Columns with $ in entry
    def convert_dollar_sign_columns(df):
        columns_with_dollar_sign = ['INCOME', 'HOME_VAL', 'BLUEBOOK',
                                    'OLDCLAIM',
                                    'CLM_AMT']

        for column in columns_with_dollar_sign:
            df[column].replace(to_replace='\D+', value='', regex=True,
                               inplace=True)
            df[column] = pd.to_numeric(df[column])
        return df

    df = convert_dollar_sign_columns(df=df)

    # End of Conversion

    # Imputation of Empty or NaN in Columns

    def convert_na_cells(colName, df, measureType):
        # Create two new column names based on original column name.
        indicatorColName = 'm_' + colName  # Tracks whether imputed.
        imputedColName = 'imp_' + colName  # Stores original & imputed data.

        # Get mean or median depending on preference.
        imputedValue = 0
        if (measureType == "median"):
            imputedValue = df[colName].median()
        elif (measureType == "mode"):
            imputedValue = float(df[colName].mode())
        else:
            imputedValue = df[colName].mean()

        # Populate new columns with data.
        imputedColumn = []
        indictorColumn = []
        for i in range(len(df)):
            isImputed = False

            # mi_OriginalName column stores imputed & original data.
            if (np.isnan(df.loc[i][colName])):
                isImputed = True
                imputedColumn.append(imputedValue)
            else:
                imputedColumn.append(df.loc[i][colName])

            # mi_OriginalName column tracks if is imputed (1) or not (0).
            if (isImputed):
                indictorColumn.append(1)
            else:
                indictorColumn.append(0)

        # Append new columns to dataframe but always keep original column.
        df[indicatorColName] = indictorColumn
        df[imputedColName] = imputedColumn
        return df

    def analysis_of_income_for_imputation(df):
        occurences_of_income = df['INCOME'].value_counts(
            dropna=False).to_dict()
        # print(occurences_of_income)
        plt.bar(["$0", "NaN"], [797, 570])
        plt.title("Occurences of Distinct Values for Income")
        plt.xlabel("Income")
        plt.ylabel("Occurences")
        plt.show()
        # Stats
        # Entries = 9732
        # Significant Non-Unique Occurences = {0: 797, "NaN": 570}
        # 1367 entries are significant non-unique. 8365 are entries remaining
        # that are relatively distinct.
        # Conclusion for imputation choice: Do not use mean or mode. Median is
        # ideal. 570 entries will be imputed.

    def analysis_of_age_for_imputation(df):
        occurences_of_age = df['AGE'].value_counts(dropna=False).to_dict()
        plt.bar(occurences_of_age.keys(), occurences_of_age.values())
        plt.title("Occurences of Distinct Values for Age")
        plt.xlabel("Ages")
        plt.ylabel("Occurences")
        plt.show()
        # The distribution of the age plot is normal and balanced.
        # Imputation with mean is reliable and only 7 entries need to be imputed so
        # the imputation will not heavily impact our regressional analysis later.

    def analysis_of_yoj_for_imputation(df):
        occurences_of_yoj = df['YOJ'].value_counts(dropna=False).to_dict()
        plt.bar(occurences_of_yoj.keys(), occurences_of_yoj.values())
        plt.title("Occurences of Distinct Values for YOJ")
        plt.xlabel("YOJ")
        plt.ylabel("Occurences")
        plt.show()
        # The distribution is not normal. There are 800, 0 value entries of the
        # 9754 entry total.
        # The majority of entries are focused around the mean.
        # There are 548 entries missing.
        # The 0 entries could be significant so testing median or mode are
        # likely more reliable than mean since there are so many 0 value entries.

    def analysis_of_home_val_for_imputation(df):
        # print(df['HOME_VAL'].value_counts(
        #     dropna=False))
        occurences_of_home_val = df['HOME_VAL'].value_counts(
            dropna=False).to_dict()
        plt.bar(["$0", "NaN"], [2908, 575])
        plt.title("Occurences of Distinct Values for Home Values")
        plt.xlabel("Home Values")
        plt.ylabel("Occurences")
        plt.show()
        # Value       Occurences
        # 0.0         2908
        # NaN          575
        # 6819 entries are not 0 or NaN
        # 575 entries are missing (NaN)
        # STD is quite high and is explainable by the occurences of $0 entries.
        # Our analysis of distinct values shows that mean and mode would be
        # invalid imputation methods for home_val. I will use median.
        #	        count	mean	std	        min	25%	50%	    75%	    max
        # HOME_VAL	9727	154523	129188.4	0	0	160661	238256	885282

    def analysis_of_car_age_for_imputation(df):
        occurences_of_car_age = df['CAR_AGE'].value_counts(
            dropna=False).to_dict()
        plt.bar(occurences_of_car_age.keys(), occurences_of_car_age.values())
        plt.title("Occurences of Distinct Values for Car Age")
        plt.xlabel("Car Age")
        plt.ylabel('Occurences')
        plt.show()
        # Distribution is not normal. There are roughly 2450 cars with an age of 1.
        # The second highest occurring age is 8 at around 650.
        # The occurences of age 1 are not likely to be erroneous.
        # Imputation with car age should be done with mean or median to find
        # the better imputation method.

    # Treat outlier in CAR_AGE column
    df.CAR_AGE = df.CAR_AGE.mask(df.CAR_AGE.lt(0), 0)

    def imputation_analysis(df):
        analysis_of_income_for_imputation(df)
        analysis_of_age_for_imputation(df)
        analysis_of_yoj_for_imputation(df)
        analysis_of_home_val_for_imputation(df)
        analysis_of_car_age_for_imputation(df)

    imputation_analysis(df)
    df = convert_na_cells("INCOME", df, "median")
    df = convert_na_cells("AGE", df, "mean")
    df = convert_na_cells("YOJ", df, "mode")
    df = convert_na_cells("HOME_VAL", df, "median")
    df = convert_na_cells("CAR_AGE", df, "mean")
    # End of Imputation

    # Dummy Variables: Handling Categorial and Ordinal/Nominal Columns
    # Treating categorical (string) information.
    df = pd.get_dummies(df, columns=['PARENT1', 'MSTATUS', 'GENDER',
                                     'EDUCATION', 'OCCUPATION', 'CAR_USE',
                                     'CAR_TYPE', 'RED_CAR', 'REVOKED',
                                     'URBANICITY'])
    # End of Dummy Variable Handling

    # Binning
    df['AGE_bin'] = pd.cut(x=df['AGE'], bins=[0, 17, 27, 37, 47, 57, 67, 77])
    tempDf = df['AGE_bin']  # Isolate columns
    # Get dummies
    dummyDf = pd.get_dummies(tempDf, columns=['AGE_bin'])
    df = pd.concat(([df, dummyDf]), axis=1)  # Join dummy df with original

    # Heatmap of Correlations of DF
    # corr = df.corr()
    # plt.subplots(figsize=(20, 15))
    # sns.heatmap(corr)
    # plt.show()

    # Separate into x and y values.

    predictors_test = ['BLUEBOOK',
                       'OLDCLAIM',
                       'CLM_FREQ',
                       'CLM_AMT',
                       'imp_INCOME',
                       'imp_YOJ',
                       'imp_HOME_VAL',
                       'imp_CAR_AGE',
                       'PARENT1_No',
                       'MSTATUS_Yes',
                       'MSTATUS_z_No',
                       'GENDER_M',
                       'GENDER_z_F',
                       'EDUCATION_<High School',
                       'EDUCATION_Bachelors',
                       'EDUCATION_Masters',
                       'EDUCATION_PhD',
                       'EDUCATION_z_High School',
                       'OCCUPATION_Clerical',
                       'OCCUPATION_Doctor',
                       'OCCUPATION_Home Maker',
                       'OCCUPATION_Lawyer',
                       'OCCUPATION_Manager',
                       'OCCUPATION_Professional',
                       'OCCUPATION_Student',
                       'OCCUPATION_z_Blue Collar',
                       'CAR_USE_Commercial',
                       'CAR_USE_Private',
                       'CAR_TYPE_Minivan',
                       'CAR_TYPE_Panel Truck',
                       'CAR_TYPE_Pickup',
                       'CAR_TYPE_Sports Car',
                       'CAR_TYPE_Van',
                       'CAR_TYPE_z_SUV',
                       'RED_CAR_no',
                       'RED_CAR_yes',
                       'REVOKED_No',
                       'REVOKED_Yes',
                       'URBANICITY_Highly Urban/ Urban',
                       'URBANICITY_z_Highly Rural/ Rural']
    X = df[predictors_test]
    y = df['CLAIM_FLAG']

    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression

    # Scale the data prior to selection.
    print("Please wait for scaling...")
    sc_x = StandardScaler()
    X_scaled = sc_x.fit_transform(X)

    print("Please wait for automated feature selection...")
    logreg = LogisticRegression(max_iter=200)
    rfe = RFE(logreg, 20)  # Select top 20 features.
    rfe = rfe.fit(X_scaled, y.values.ravel())
    print("Feature selection is complete.")

    def getSelectedColumns(ranking):
        # Extract selected indices from ranking.
        indices = []
        for i in range(0, len(ranking)):
            if (ranking[i] == 1):
                indices.append(i)
        # Build list of selected column names.
        counter = 0
        selectedColumns = []
        for col in X:
            if (counter in indices):
                selectedColumns.append(col)
            counter += 1
        return selectedColumns

    selectedPredictorNames = getSelectedColumns(rfe.ranking_)

    # Show selected names from RFE.
    print("\n*** Selected names: ")
    for i in range(0, len(selectedPredictorNames)):
        print(selectedPredictorNames[i])

    # prepare cross validation with three folds and 1 as a random seed.
    # Separate into x and y values.
    count = 0
    kfold = KFold(3, True, 1)
    for train, test in kfold.split(df[selectedPredictorNames]):
        X = df[selectedPredictorNames]
        y = df[['CLAIM_FLAG']]
        X_train = X.iloc[train, :]  # Gets all rows with train indexes.
        y_train = y.iloc[train, :]
        X_test = X.iloc[test, :]
        y_test = y.iloc[test, :]
        X_scaled = sc_x.fit_transform(X)

        # Perform logistic regression.
        logisticModel = LogisticRegression(fit_intercept=True, random_state=0,
                                           solver='liblinear')
        # Fit the model.
        logisticModel.fit(X_train, np.ravel(y_train))

        y_pred = logisticModel.predict(X_test)
        y_prob = logisticModel.predict_proba(X_test)



        # Show chi-square scores for each feature.
        # There is 1 degree freedom since 1 predictor during feature evaluation.
        # Generally, >=3.8 is good)

        test = SelectKBest(score_func=chi2, k=20)
        XScaled = MinMaxScaler().fit_transform(X)
        chiScores = test.fit(XScaled, y)  # Summarize scores
        np.set_printoptions(precision=3)

        # Search here for insignificant features.
        print("\nPredictor Chi-Square Scores: " + str(chiScores.scores_))

        # Split data.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0)

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
        print("\n***K-fold: " + str(count))
        print('\nAccuracy: ', metrics.accuracy_score(y_test, y_pred))
        print("\nConfusion Matrix")
        print(cm)

        from sklearn.metrics import classification_report, roc_auc_score

        print(classification_report(y_test, y_pred))

        from sklearn.metrics import average_precision_score
        average_precision = average_precision_score(y_test, y_pred)

        print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))

        # calculate scores
        auc = roc_auc_score(y_test, y_prob[:, 1], )
        print('Logistic: ROC AUC=%.3f' % (auc))

        # Stat Summary: accuracy, precision, recall, f1 scores along with averages and
        # standard deviations of these scores for all folds.
        # Show model coefficients and intercept.
        print(f"\nStatistical Summary of Model {number}")
        print("\nModel Intercept: ")
        print(logisticModel.intercept_)
        print("\nModel Coefficients: ")
        print(logisticModel.coef_)
        # Prediction with test data
        pred = logisticModel.predict(X_test)
        # Show stats about the regression.
        mse = mean_squared_error(y_test, pred)
        RMSE = np.sqrt(mse)
        print("\nRMSE: " + str(RMSE))
        print("\nr2_score", r2_score(y_test, pred))

        # ROC CURVE CHART, and CUMUL GAINS CHART
        def create_roc_curve_chart_and_cumul_chart():
            # calculate roc curves chart
            CUT_OFF = 0.50
            lr_fpr, lr_tpr, _ = roc_curve(y_test, y_prob[:, 1])
            plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
            plt.plot([0, 1], [0, 1], '--', label=f"CUT-OFF{CUT_OFF}")
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title("ROC CURVE")
            plt.legend()
            plt.show()
            # cumulative gains chart
            clf = LogisticRegression(
                random_state=0, multi_class='multinomial', solver='newton-cg')

            clf.fit(X_train, y_train)
            predicted_probas = clf.predict_proba(X_test)
            y_pred = clf.predict(X_test);
            import scikitplot as skplt
            skplt.metrics.plot_cumulative_gain(y_test, predicted_probas)
            skplt.metrics.plot_lift_curve(y_test, predicted_probas)
            plt.show()
        y_pred = create_roc_curve_chart_and_cumul_chart()
        print(f"\nEnd of Model {number}")


if __name__ == '__main__':
    main()
