import numpy as np
import seaborn as sns
sns.set(style='whitegrid')
import pandas as pd
import matplotlib.pyplot as plt
#
# df = pd.read_csv('../Data/final_data.csv')
# #
# df = df[df['Category'] == 'Table']
# df.drop(['Item Code', 'Category', 'Type', 'Material'], inplace=True, axis=1)
# #
# cpi_gdp_df = pd.read_csv('../Data/cpi_gdp_data_rate.csv', usecols=['Date', 'GDP', 'CPI'])
#
# for d_index, d_row in df.iterrows():
#     for c_index, c_row in cpi_gdp_df.iterrows():
#         if df['Date'][d_index] == cpi_gdp_df['Date'][c_index]:
#             df['CPI'][d_index] = c_row['CPI']
#             df['GDP'][d_index] = c_row['GDP']
#
# print df
# df.to_csv('../Data/table_data_rnn.csv')

df = pd.read_csv('../Data/table_data_rnn.csv')
df[['Cost', 'GDP', 'Quantity']] = (
                                        df[['Cost', 'GDP', 'Quantity']] -
                                        df[['Cost', 'GDP', 'Quantity']].min())/\
                                        (df[['Cost', 'GDP', 'Quantity']].max()-
                                        df[['Cost', 'GDP', 'Quantity']].min()
                                        )
df.loc[df['CPI'] == df['CPI'].max(), 'CPI'] = 2
# Normalising CPI between -1 and 1
df['CPI'] = 2*((df['CPI'] - df['CPI'].min())/(df['CPI'].max() - df['CPI'].min())) - 1


# df['CPI'] = df['CPI'] / 10.
df.drop(['Unnamed: 0'], inplace=True, axis=1)
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)
df.drop(['GDP', 'Cost'], inplace=True, axis=True)
# print df
df.plot()
plt.show()