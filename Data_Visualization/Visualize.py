import numpy as np
import seaborn as sns
sns.set(style='whitegrid')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('../Data/final_data.csv')
# df.sort_index(inplace=True)
# df.index = pd.to_datetime(df.index)

# plot_df = df.loc[df['Category']=='Cupboard'].groupby(['Date']).sum()

plot_df = df.loc[df['Type'] == 'Coffee Table'].groupby(['Date', 'GDP', 'CPI']).sum()

# print plot_df

plot_df.reset_index(level=['GDP', 'CPI'], inplace=True)
plot_df.reset_index(inplace=True)

# categories = df['Category'].unique()
# df['Type'].fillna('NAN', inplace=True)
# with open('./category_breakdown.txt', 'a+') as f:
#     f.write("BREAKDOWN OF CATEGORIES: ")
#     for category in categories:
#         f.write('\n\n{}: '.format(category))
#         type_set = set()
#         for _, row in df.iterrows():
#             print row['Category']
#             if row['Category'] == category:
#                 if row['Type'] not in type_set:
#                     f.write('{}({}), '.format(row['Type'],
#                                               df['Type'].loc[(df['Type'] == row['Type']) &
#                                                              (df['Category'] == category)].count()))
#                     type_set.add(row['Type'])
#
#


# print plot_df
# Calculating the CPI difference so the rate is calculated. The values denote a trend in the increase and decrease of
# the spending capability.
#
# for s_index, s_row in plot_df.iterrows():
#     if s_index == 94:
#         plot_df['CPI'][s_index] = plot_df['CPI'][s_index-1]
#         break
#     value = plot_df['CPI'][s_index + 1] - plot_df['CPI'][s_index]
#     plot_df['CPI'][s_index] = value
#
# plot_df.loc[(plot_df['CPI'] == 172) | (plot_df['CPI'] == 286), 'CPI'] = 2
# print plot_df
# Likewise for GDP. Since delta GDP values are not negative, the values denote how much or how less it has increased
# from previous months

# for s_index, s_row in plot_df.iterrows():
#     if s_index == 94:
#         plot_df['GDP'][s_index] = plot_df['GDP'][s_index-1]
#         break
#     value = plot_df['GDP'][s_index + 1] - plot_df['GDP'][s_index]
#     plot_df['GDP'][s_index] = value

# print plot_df['CPI']
# Normalising them between 0 and 1

# plot_df.to_csv('../Data/cpi_gdp_data_rate.csv')
# plot_df[['Quantity', 'Cost']] = (
#                                         plot_df[['Quantity', 'Cost']] -
#                                         plot_df[['Quantity', 'Cost']].min())/\
#                                         (plot_df[['Quantity', 'Cost']].max()-
#                                         plot_df[['Quantity', 'Cost']].min()
#                                         )
# Normalising CPI between -1 and 1
# plot_df['CPI'] = 2*((plot_df['CPI'] - plot_df['CPI'].min())/(plot_df['CPI'].max() - plot_df['CPI'].min())) - 1
scaler = MinMaxScaler()
train_temp_scaled = scaler.fit_transform(plot_df[['Quantity', 'Cost', 'CPI']])
print train_temp_scaled
plot_df[['Quantity', 'Cost', 'CPI']] = train_temp_scaled
plot_df.set_index('Date', inplace=True)
plot_df.index = pd.to_datetime(plot_df.index)
plot_df.drop(['GDP','CPI', 'Cost'], inplace=True, axis=1)
# print plot_df
plot_df[(plot_df.index >= '2010-01-01') & (plot_df.index <= '2017-12-01')].plot()
plt.show()