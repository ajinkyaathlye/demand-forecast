import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------
# Define required columns and appropriate dtypes
# ---------------------------------------------------------

sales_cols = ['Item Code', 'Date', 'Quantity', 'Cost']
gdp_cols = ['Year/ Industry', 'Quarter', 'Total Gross Value Added at Basic Price']
inventory_cols = ['Item Code', 'Category', 'Type', 'Material']
cpi_cols = ['Date', 'Consumer Price Index for Industrial Workers (2001=100)']


gdp_df = pd.read_csv(
    '../Data/GDP_quarterly_RBI.csv',
    usecols=gdp_cols,
    dtype={'Year/ Industry':np.string_}
)

cpi_df = pd.read_csv(
    '../Data/RBI_Major_Price_Indices_CPI.csv',
    usecols=cpi_cols,
    parse_dates=['Date']
)

furniture_df = pd.read_csv(
    '../Data/Sales_Furniture.csv',
    usecols=sales_cols,
    parse_dates=['Date']
)
home_decor_df = pd.read_csv(
    '../Data/Sales_Home_Decor.csv',
    usecols=sales_cols,
    parse_dates=['Date']
)
inventory_df = pd.read_csv(
    '../Data/complete_sales_added_inventory.csv',
    usecols=inventory_cols
)

# Initialise new, final df
df = pd.DataFrame()

"""
# ---------------------------------------------------------
# Process furniture_df
# ---------------------------------------------------------
furniture_df['Item Code'] = furniture_df['Item Code'].apply(
    lambda x: x.split()[0]
)
# Check if any item code is compromised after the above logic
furniture_df['Item Code'].isnull().sum()
# Above value = 0 --> all good

# Start appending columns in final df
df['Item Code'] = furniture_df['Item Code']
df['Date'] = furniture_df['Date']
df['Quantity'] = furniture_df['Quantity']
df['Cost'] = furniture_df['Cost']

# ---------------------------------------------------------
# Process home_decor_df
# ---------------------------------------------------------
home_decor_df['Item Code'] = home_decor_df['Item Code'].apply(
    lambda x: x.split()[0]
)

home_decor_df['Item Code'].isnull().sum() # equal to 0

# Merge the two dfs based on the item code

df = pd.concat([furniture_df, home_decor_df], ignore_index=True, sort=False)

# print df
# Sales data is prepared

# ---------------------------------------------------------
# Map inventory data with sales data
# ---------------------------------------------------------

df['Category'] = np.nan
df['Type'] = np.nan
df['Material'] = np.nan

inventory_df.dropna(how='all', inplace=True)

inventory_df['Item Code'] = inventory_df['Item Code'].apply(
    lambda x: x.split()[0]
)

df = df.sort_values('Item Code')
inventory_df = inventory_df.sort_values('Item Code')

for s_index, s_row in df.iterrows():
    flag = 0
    for i_index, i_row in inventory_df.iterrows():
        if s_row['Item Code'] == i_row['Item Code']:
            df['Category'][s_index] = inventory_df['Category'][i_index]
            df['Type'][s_index] = inventory_df['Type'][i_index]
            df['Material'][s_index] = inventory_df['Material'][i_index]
            print df['Item Code'][s_index]
            flag = 1
        if flag == 1:
            break

df.to_csv('../Data/sales_and_inventory.csv')

print df

"""

"""

# ---------------------------------------------------------
# Add gdp data
# ---------------------------------------------------------
# Data is split up in quarters --> Divide equally for months in a year and assign values

# Import the sales and inventory data
df = pd.read_csv('../Data/sales_and_inventory.csv', parse_dates=['Date'])
df['Date'] = df['Date'].apply(lambda x: x.strftime('%Y-%m'))
gdp_df['Year/ Industry'].fillna(method='ffill', inplace=True)


# Import mappings
stream = file('./mappings.yaml', 'r')
mappings = yaml.load(stream)

# Add GDP column
df['GDP'] = np.nan
quarters = mappings['gdp_quarter_to_date']

for s_index, s_row in df.iterrows():
    # if s_row['Date'][:4] == str(g_row['Year/ Industry'])[:4]:
    for quarter, months in quarters.iteritems():
        for month in months:
            if s_row['Date'][5:7] == month:
                df['GDP'][s_index] = gdp_df.loc[
                   (gdp_df['Year/ Industry'] == s_row['Date'][:4])
                   & (gdp_df['Quarter'] == quarter), 'Total Gross Value Added at Basic Price'
                    ].values[0]


print df
df.to_csv('../Data/GPD_inclusive_data.csv')
"""

"""
# ---------------------------------------------------------
# Add cpi data
# ---------------------------------------------------------

df = pd.read_csv('../Data/GPD_inclusive_data.csv', parse_dates=['Date'])
# Add the CPI column to d
df['CPI'] = np.nan
# Modify date to match the date format in df
cpi_df['Date'] = cpi_df['Date'].apply(lambda x: x.strftime('%Y-%d'))
df['Date'] = df['Date'].apply(lambda x: x.strftime('%Y-%m'))

for s_index, s_row in df.iterrows():
    df['CPI'][s_index] = cpi_df.loc[cpi_df['Date'] == s_row['Date'],
                              'Consumer Price Index for Industrial Workers (2001=100)'].values[0]

print df

df.to_csv('../Data/final_data.csv')

# df['Date'] = pd.date_range(start='1/1/2010', end='12/31/2017', freq='M')
"""

df = pd.read_csv('../Data/final_data.csv')
df = pd.concat( [ pd.get_dummies( df[ x ] ) for x in cols ], axis=1 ).assign( target=df.target )