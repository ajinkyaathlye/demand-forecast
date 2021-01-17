import pandas as pd
import numpy as np

df = pd.read_csv('../Data/final_data.csv')

# Convert the dates into months for now
df['Date'] = df['Date'].apply(lambda x: x[5:7])
df['Date'] = df['Date'].astype(np.int16)


# ------------------------------------------------------
# Categorise all as 1-H-E; all unique entries will have a column of their own
# ------------------------------------------------------

list_of_columns = ['Date', 'Category', 'Type', 'Material']

for column in list_of_columns:
    values = df[column].unique()
    values.sort()
    # Initialise the columns to 0
    for value in values:
        df[str(value)] = 0

    for value in values:
        df[str(value)] = df[column].apply(lambda x: 1 if x == value else 0)

    df.drop([column], axis=1, inplace=True)


# for column in list_of_columns:
#     values = df[column].unique()
#     values.sort()
#     dict={}
#
#     for i, value in enumerate(values):
#         dict[value] = i+1
#
#     df[column] = df[column].apply(lambda x: dict[x])
#
# df.to_csv('../Data/categorised_data_2.csv')
