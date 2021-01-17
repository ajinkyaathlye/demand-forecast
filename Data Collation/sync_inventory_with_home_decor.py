import pandas as pd
import yaml

# Required columns from csv files
sales_cols = ['Item Code', 'Category', 'Date', 'Quantity', 'Cost']
inventory_cols = ['Item Code', 'Category', 'Type', 'Material']

# Import mappings
stream = file('./mappings.yaml', 'r')
mappings = yaml.load(stream)

# Import data
home_decor_df = pd.read_csv(
    '../Data/Sales_Home_Decor.csv',
    usecols=sales_cols,
    parse_dates=['Date']
)
inventory_df = pd.read_csv(
    '../Data/furniture_sales_added_inventory.csv',
    usecols=inventory_cols
)
"""
# ---------------------------------------------------------
# Remove common entries of inventory_df and home_decor_df from home_decor_df
# ---------------------------------------------------------

# Remove null rows from the inventory data
inventory_df.dropna(how='all', inplace=True)

# Remove the description from the Item Code columns. Retain description for processing purposes in furniture_df
inventory_df['Item Code'] = inventory_df['Item Code'].apply(
    lambda x: x.split()[0]
)
home_decor_df['Item Code#2'] = home_decor_df['Item Code'].apply(
    lambda x: x.split()[0]
)

for hd_value in home_decor_df['Item Code#2']:
    for inv_value in inventory_df['Item Code']:
        if inv_value == hd_value:
            home_decor_df.loc[home_decor_df['Item Code#2'] == inv_value, 'Item Code#2'] = np.nan

print len(home_decor_df['Item Code#2'].unique()) # equals 573

# ---------------------------------------------------------
# Remove the rows for which Item Code is empty
# ---------------------------------------------------------
home_decor_df.dropna(inplace=True)
home_decor_df.reset_index(drop=True, inplace=True)
home_decor_df.drop_duplicates(subset='Item Code', inplace=True)
# Check which categories need re-population in the unique values
for category in home_decor_df['Category'].unique():
    print category + ": " + str(home_decor_df.loc[home_decor_df['Category'] == category, 'Category'].count())

print "==============================================================="

for type in inventory_df.loc[inventory_df['Category'] == 'Home Decor', 'Type'].unique():
    print type + ": " + inventory_df.loc[inventory_df['Type'] == type, 'Type'].count().astype(np.string_)


# ---------------------------------------------------------
# Modify the df unique codes with their category, type, and material
# ---------------------------------------------------------

# Inserting values according to mappings in mappings.yaml
home_decor_df['new_category'] = home_decor_df['Category'].apply(lambda x: mappings['home_decor'][x]['Category'])

home_decor_df['new_type'] = home_decor_df['Category'].apply(
    lambda x: mappings['home_decor'][x]['Type'] if mappings['home_decor'][x]['Type'] != 'NULL' else np.nan
)
home_decor_df['new_material'] = home_decor_df['Category'].apply(
    lambda x: mappings['home_decor'][x]['Material'] if mappings['home_decor'][x]['Material'] != 'NULL' else np.nan
)


home_decor_df.to_csv('../Data/home_decor_unique_dataset.csv', columns=['Item Code', 'new_category', 'new_type', 'new_material'])

# Data not modified yet

# ---------------------------------------------------------
# Add the extra entries from the home decor sales to inventory data
# ---------------------------------------------------------
"""

modified_data = pd.read_csv('../Data/home_decor_unique_dataset.csv', usecols=[
    'Item Code',
    'Category',
    'Type',
    'Material'
])

inventory_df = pd.concat([inventory_df, modified_data], ignore_index=True, sort=False)
print inventory_df

# ---------------------------------------------------------
# Save to csv
# ---------------------------------------------------------

inventory_df.to_csv('../Data/complete_sales_added_inventory.csv')
