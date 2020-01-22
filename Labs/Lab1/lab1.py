"""
Lab 1 COMP 3948 Predictive Modelling

@author Aiman Ismail
@date January 8, 2020
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def exercise_one():
    text_two = "She sells seashells by the sea shore."
    index_of_list_of_sea = [i for i in range(len(text_two)) if text_two.find(
        'sea', i) == i]
    print("sea is at", index_of_list_of_sea)


def exercise_two():
    text = "A lazy dog jumped over a log."
    new_text = text.replace("lazy", "\b")
    print(new_text)


def exercise_three():
    full_name = "Bob Jones"
    space_position = full_name.find(" ")
    last_name = full_name[space_position + 1:]
    print(last_name)


def exercise_four():
    sentence_array = ['A', 'lazy', 'dog', 'jumped', 'over', 'a', 'log.']
    delimiter = ','
    new_setence = delimiter.join(sentence_array)
    print(new_setence)


def exercise_five():
    dataSet = {'First Name': ['Jonny', 'Nira', 'Holly'],
               'Last Name': ['Staub', 'Arora', 'Conway'],
               'Grades': [85, 95, 91]}
    dataFrame = pd.DataFrame(dataSet, columns=['First Name',
                                               'Last Name',
                                               'Grades'])
    print(dataFrame)


def exercise_six():
    # Create data set.
    data_set = {'Market': ['S&P 500', 'Dow', 'Nikkei'],
                'Last': [2932.05, 26485.01, 21087.16]}

    # Create dataframe with data set and named columns.
    df = pd.DataFrame(data_set, columns=['Market', 'Last'])

    # Show original DataFrame.
    print("\n*** Original DataFrame ***")
    print(df)

    # Create change column.
    change = [-21.51, -98.41, -453.83]
    percentage_change = [-0.73, -0.37, -2.11]

    # Append change column.
    df['Change'] = change
    df['Percentage Change'] = percentage_change

    # Show revised DataFrame.
    print("\n*** Adjusted DataFrame ***")
    print(df)


def exercise_seven():
    # Create data set.
    data_set = {'Market': ['S&P 500', 'Dow', 'Nikkei'],
                'Last': [2932.05, 26485.01, 21087.16]}

    # Create dataframe with data set and named columns.
    df = pd.DataFrame(data_set, columns=['Market', 'Last'])

    # Show original DataFrame.
    print("\n*** Original DataFrame ***")
    print(df)

    # Add new line.
    print("\n")

    # Show names only
    for i in range(len(df)):
        print(df.loc[i]['Last'])


def exercise_eight():
    data_set = {'Market': ['S&P 500', 'Dow', 'Nikkei'],
                'Last': [2932.05, 26485.01, 21087.16]}

    # Create dataframe with data set and named columns.
    df = pd.DataFrame(data_set, columns=['Market', 'Last'])

    # Show original DataFrame.
    print("\n*** Original DataFrame ***")
    print(df)

    # Add new line.
    print("\n")

    # Show first row only.
    print(f"{df.loc[0]['Market']} {df.loc[0]['Last']}")


def exercise_nine():
    # Create data set.
    data_set = {'Market': ['S&P 500', 'Dow', 'Nikkei'],
                'Last': [2932.05, 26485.01, 21087.16]}

    # Create data frame with data set and named columns.
    df1 = pd.DataFrame(data_set, columns=['Market', 'Last'])

    # Show original DataFrame.
    print("\n*** Original DataFrame ***")
    print(df1)

    dataSet2 = {'Market': ['Hang Seng', 'DAX'],
                'Last': [26918.58, 11872.44]}
    df2 = pd.DataFrame(dataSet2, columns=['Market', 'Last'])

    data_set_three = {'Market': ['FTSE100'], 'Last': 7407.06}
    df3 = pd.DataFrame(data_set_three, columns=['Market', 'Last'])

    df1 = df1.append(df2)
    df1 = df1.append(df3)
    print("\n*** Adjusted DataFrame ***")
    print(df1)


def exercise_ten():
    data_set = {'Market': ['S&P 500', 'Dow'], 'Last': [2932.05, 26485.01]}
    stock_dictionary = {'Market': 'Nikkei', 'Last': 21087.16}
    df = pd.DataFrame(data_set, columns=['Market', 'Last'])
    print("\n **Original Data Frame ***")
    df = df.append(stock_dictionary, ignore_index=True)
    print(df);
    dax_dictionary = {'Market': 'Dax', 'Last': 11872.44}
    print("\n ***Adjusted Data Frame for Dax")
    df = df.append(dax_dictionary, ignore_index=True)
    print(df)


def exercise_eleven():
    path = "./bodyfat.txt"
    df = pd.read_table(path, skiprows=1,
                       sep='\t',
                       names=('Density', 'Pct.BF', 'Age', 'Weight', 'Height',
                              'Neck', 'Chest', 'Abdomen', 'Waist', 'Hip',
                              'Thigh',
                              'Ankle', 'Knee', 'Bicep', 'Forearm', 'Wrist'))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    # print("First two rows")
    # print(df.head(2))
    # print("Last two rows")
    # print(df.tail(2))
    # print(df.dtypes)
    # print(df.describe())
    print(df.describe().round(2))


def exercise_twelve():
    path = "./bodyfat.txt"
    df = pd.read_table(path, skiprows=1,
                       sep='\t',
                       names=('Density', 'Pct.BF', 'Age', 'Weight', 'Height',
                              'Neck', 'Chest', 'Abdomen', 'Waist', 'Hip',
                              'Thigh',
                              'Ankle', 'Knee', 'Bicep', 'Forearm', 'Wrist'))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    df2 = df[['Height', 'Waist', 'Weight', 'Pct.BF']]
    print(df2)


def exercise_thirteen():
    path = "./bodyfat.txt"
    df = pd.read_table(path, skiprows=1,
                       sep='\t',
                       names=('Density', 'Pct.BF', 'Age', 'Weight', 'Height',
                              'Neck', 'Chest', 'Abdomen', 'Waist', 'Hip',
                              'Thigh',
                              'Ankle', 'Knee', 'Bicep', 'Forearm', 'Wrist'))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    df2 = df[['Height', 'Waist', 'Weight', 'Pct.BF']]
    df2 = df2.rename({'Pct.BF': 'Percent Body Fat'}, axis=1)
    print(df2)


def exercise_fourteen():
    import pandas as pd

    # Import data into a DataFrame.
    path = "babysamp-98.txt"
    df = pd.read_table(path, skiprows=1,
                       delim_whitespace=True,
                       names=(
                           'MomAge', 'DadAge', 'MomEduc', 'MomMarital',
                           'numlive',
                           "dobmm", 'gestation', 'sex', 'weight',
                           'prenatalstart',
                           'orig.id', 'preemie'))
    # Show all columns.
    pd.set_option('display.max_columns', None)

    # Increase number of columns that display on one line.
    pd.set_option('display.width', 1000)

    # print("\n FIRST 5 COLUMNS")  # Prints title with space before.
    # print(df.head(2))
    #
    # print("\n LAST 3 COLUMNS")
    # print(df.head(2))
    #
    # # Show data types for each columns.
    # print("\n DATA TYPES")  # Prints title with space before.
    # print(df.dtypes)
    #
    # # Show statistical summaries for numeric columns.
    # print("\nSTATISTIC SUMMARIES for NUMERIC COLUMNS")
    # print(df.describe())

    # # Show summaries for objects like dates and strings.
    # print("\nSTATISTIC SUMMARIES for DATE and STRING COLUMNS")
    # print(df.describe(include=['object']))
    #
    # print("\nTOP FREQUENCY FIRST")
    # print(df['MomAge'].value_counts(), end='')
    # print("\nLOWEST FREQUENCY FIRST", end='')
    # print(df['MomAge'].value_counts(ascending=True), end='')
    # print("\nFREQUENCY SORTED by MOTHER AGE", end='')
    # print(df['MomAge'].value_counts().sort_index(), end='')

    print("\nFREQUENCY COUNT OF UNIQUE VALUES")
    print(df['MomEduc'].value_counts(normalize=True).sort_index(
        ascending=False), end='')


def exercise_fifteen():
    # Import data into a DataFrame.
    path = "babysamp-98.txt"
    df = pd.read_csv(path, skiprows=1,
                     sep='\t',
                     names=(
                         'MomAge', 'DadAge', 'MomEduc', 'MomMarital',
                         'numlive',
                         "dobmm", 'gestation', 'sex', 'weight',
                         'prenatalstart',
                         'orig.id', 'preemie'))

    # Rename the columns so they are more reader-friendly.
    df = df.rename({'MomAge': 'Mom Age', 'DadAge': 'Dad Age',
                    'MomEduc': 'Mom Edu', 'weight': 'Weight'},
                   axis=1)  # new method
    # Show all columns.
    pd.set_option('display.max_columns', None)

    # Increase number of columns that display on one line.
    pd.set_option('display.width', 1000)
    print("Count:", df['Mom Age'].count())
    print("Min:", df['Mom Age'].min())
    print("Max:", df['Mom Age'].max())
    print("Mean:", df['Mom Age'].mean())
    print("Median:", df['Mom Age'].median())
    print("Standard Deviation:", df['Mom Age'].std())


def exercise_sixteen():
    # The data file path and file name need to be configured.
    # PATH = "/Users/pm/Desktop/DayDocs/2019_2020/PythonForDataAnalytics/workingData/"
    CSV_DATA = "phone_data.csv"

    # Note this has a comma separator.
    df = pd.read_csv(CSV_DATA, skiprows=1, encoding="ISO-8859-1",
                     sep=',',
                     names=(
                         'index', 'date', 'duration', 'item', 'month',
                         'network',
                         'network_type'))

    # Get count of items per month.
    dfStats = df.groupby('network')['index'] \
        .count().reset_index().rename(columns={'index': '# Calls'})

    # Get duration mean for network groups and convert to DataFrame.
    dfDurationMean = df.groupby('network')['duration'] \
        .mean().reset_index().rename(columns={'duration': 'Duration '
                                                          'Mean'})

    # Get duration max for network groups and convert to DataFrame.
    dfDurationMax = df.groupby('network')['duration'] \
        .max().reset_index().rename(columns={'duration': 'Duration Max'})

    # Append duration mean to stats matrix.
    dfStats['Duration Mean'] = dfDurationMean['Duration Mean']

    # Append duration max to stats matrix.
    dfStats['Duration Max'] = dfDurationMax['Duration Max']
    print(dfStats)


def main():
    exercise_sixteen()


if __name__ == '__main__':
    main()
