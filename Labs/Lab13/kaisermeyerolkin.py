import pandas as pd
import numpy  as np

# Can get from PyCharm terminal.
from factor_analyzer import FactorAnalyzer

# Read data.
from sklearn.linear_model import LinearRegression
PATH        = "../dataset/"
CSV_DATA    = "Factor-Hair-Revised.csv"
data        = pd.read_csv(PATH + CSV_DATA, sep=',')

# Create data frame without ID and Satisfaction columns.
df = data.copy()
del df['ID']
del df['Satisfaction']

# Display all columns of the data frame.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head(2))

# Bartlett's test of sphericity tests the hypothesis that your correlation matrix
# is an identity matrix. If the correlation matrix is an identity matrix the
# columns are unrelated and are therefore unsuitable for structure detection.
# If the test is insignificant do not use factor analysis.
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value, p_value=calculate_bartlett_sphericity(df)

print("\nBartlett's test chi-square value: ")
print(chi_square_value)

print("\nBartlett's test p-value: ")
print(p_value)

# Kaiser-Meyer-Olkin (KMO) test measures the proportion of variance among
# variables that might be common variance. The lower the proportion,
# the more suited your data is to Factor Analysis. Factor analysis is suitable
# for scores of 0.6 (and sometimes 0.5) and above.
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(df)
print("\nKaiser-Meyer-Olkin (KMO) Test: Suitability of data for factor analysis.")
print(kmo_model)

# Create factor analysis and examine loading vectors and Eigenvalues.
fa = FactorAnalyzer(rotation=None)
fa.fit(df)
print("\nFactors:")
print(fa.loadings_)

ev, v = fa.get_eigenvalues()
print("\nEignenvalues:")
print(ev)
