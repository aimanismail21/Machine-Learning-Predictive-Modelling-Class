import pandas as pd
from   sqlalchemy import create_engine

# The data file path and file name need to be configured.
PATH     = "../dataset/"

# Create the database at the specified path.
DB_FILE    = 'ramenReviews.db'

engine     = create_engine('sqlite:///' + PATH + DB_FILE, echo=False)
connection = engine.connect()

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

# Placed query in this function to enable code re-usuability.
def showQueryResult(sql):
    print("\n*** Showing SQL statement")
    print(sql)
    # Perform query
    subDf = pd.read_sql(sql, connection)
    print("\n*** Showing dataframe summary")
    return subDf

# Get ramen reviews.
sql    = "SELECT Brand, count(Brand) FROM Review GROUP BY Brand"
print(showQueryResult(sql).head())
