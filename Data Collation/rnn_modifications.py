import numpy as np
import pandas as pd

# final_data.csv is assumed to be manually modified to have numeric cost data and removal of the entry with the null
# value (921 or something)
df = pd.read_csv('../Data/final_data.csv')
df = df.loc[df['Category'] == 'Table'].groupby(['Date', 'GDP', 'CPI']).sum()

df.reset_index(level=['GDP', 'CPI'], inplace=True)
df.reset_index(inplace=True)

for s_index, s_row in df.iterrows():
    if s_index == 94:
        df['CPI'][s_index] = df['CPI'][s_index-1]
        break
    value = df['CPI'][s_index + 1] - df['CPI'][s_index]
    df['CPI'][s_index] = value


for s_index, s_row in df.iterrows():
    if s_index == 94:
        df['GDP'][s_index] = df['GDP'][s_index-1]
        break
    value = df['GDP'][s_index + 1] - df['GDP'][s_index]
    df['GDP'][s_index] = value