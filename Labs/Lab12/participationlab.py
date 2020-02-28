from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas                   as pd
import numpy                    as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold

# Read Brix value data.
CSV_DATA = "peach_spectra_brixvalues.csv"
df = pd.read_csv(CSV_DATA, sep=',')

# Split the data using k-fold validation.
kfold = KFold(4, True)  # 4-splits, shuffle randomly.
foldCount = 0
for train, test in kfold.split(df):
    print("\n\n******* Fold Count: " + str(foldCount) + " *******")
    foldCount += 1
    print(df.describe())

    # Extract train and test values.
    dfTrain = df.iloc[train, :]
    dfTest = df.iloc[test, :]

    y_train = dfTrain['Brix']
    y_test = dfTest['Brix']

    dfXTrain = dfTrain.copy()  # Copy to avoid affecting original df.
    del dfXTrain['Brix']  # Drop Brix column.
    dfXTest = dfTest.copy()  # Copy to avoid affecting original df.
    del dfXTest['Brix']  # Drop Brix column.
    text = "wl"
    # for i in range(5, 601):
    #     del dfXTrain[text + str(i)]
    #     del dfXTest[text + str(i)]


    # Scale X values.
    scaler = StandardScaler()
    Xtrain_scaled = scaler.fit_transform(dfXTrain)
    Xtest_scaled = scaler.fit_transform(dfXTest)

    # Generate PCA components.
    pca = PCA(n_components=5)  # Adjust number of components with n_components=

    # Always fit PCA with train data. Then transform the train data.
    X_reduced_train = pca.fit_transform(Xtrain_scaled)

    # Transform test data with PCA
    X_reduced_test = pca.transform(Xtest_scaled)

    print("\nPrincipal Components")
    print(pca.components_)

    print("\nExplained variance: ")
    print(pca.explained_variance_)

    # Train regression model on training data
    model = LinearRegression()
    model.fit(X_reduced_train, y_train)

    # Predict with test data.
    pred = model.predict(X_reduced_test)

    # Show stats about the regression.
    mse = mean_squared_error(y_test, pred)
    RMSE = np.sqrt(mse)
    print("\nRMSE: " + str(RMSE))

    '''
    print("\nModel Coefficients")
    print(model.coef_)

    print("\nModel Intercept")
    print(model.intercept_)
    '''

    from sklearn.metrics import r2_score

    print("\nr2_score", r2_score(y_test, pred))

    # For each principal component, calculate the VIF and save in dataframe
    vif = pd.DataFrame()

    # Show the VIF score for the principal components.
    vif["VIF Factor"] = [variance_inflation_factor(X_reduced_train, i) \
                         for i in range(X_reduced_train.shape[1])]
    print(vif["VIF Factor"])


    # Show the scree plot.
    import matplotlib.pyplot as plt

    # cor_mat = np.corrcoef(np.transpose(X_reduced_train))
    # cov_mat = np.cov(np.transpose(X_reduced_train))
    # eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    list_of_num = [num for num in range(1, len(pca.explained_variance_) + 1)]
    eig_vals = pca.explained_variance_
    plt.plot(list_of_num, eig_vals, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.show()

    # Calculate cumulative values.
    sumEigenvalues = eig_vals.sum()
    cumulativeValues = []
    cumulativeSum = 0
    for i in range(0, len(eig_vals) + 1):
        cumulativeValues.append(cumulativeSum)
        if (i < len(eig_vals)):
            cumulativeSum += eig_vals[i] / sumEigenvalues

    # Show cumulative variance plot.
    import matplotlib.pyplot as plt
    list_of_vals_start_zero = [num for num in range(0,
                                                    len(
                                                        pca.explained_variance_)+1)]
    plt.plot(list_of_vals_start_zero, cumulativeValues, 'ro-', linewidth=2)
    plt.title('Variance Explained by Principal Components')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.show()


