import pandas as pd
import yaml

# Required columns from csv files
sales_cols = ['Item Code', 'Category', 'Date', 'Quantity', 'Cost']
inventory_cols = ['Item Code', 'Category', 'Type', 'Material']

# Import mappings
stream = file('./mappings.yaml', 'r')
mappings = yaml.load(stream)

# Import data
furniture_df = pd.read_csv(
    '../Data/Sales_Furniture.csv',
    usecols=sales_cols,
    parse_dates=['Date']
)
inventory_df = pd.read_csv(
    '../Data/Inventory_Details_Report_XL.csv',
    usecols=inventory_cols
)
"""
# ---------------------------------------------------------
# Remove common entries of inventory_df and furniture_df from furniture_df
# ---------------------------------------------------------

# Remove null rows from the inventory data
inventory_df.dropna(how='all', inplace=True)

# Maintain a copy of item code with descriptions
inventory_df['Item Code Temp'] = inventory_df['Item Code']

# Remove the description from the Item Code columns. Retain description for processing purposes in furniture_df
inventory_df['Item Code'] = inventory_df['Item Code'].apply(
    lambda x: x.split()[0]
)
furniture_df['Item Code#2'] = furniture_df['Item Code'].apply(
    lambda x: x.split()[0]
)

for fur_value in furniture_df['Item Code#2']:
    for inv_value in inventory_df['Item Code']:
        if inv_value == fur_value:
            furniture_df.loc[furniture_df['Item Code#2'] == inv_value, 'Item Code#2'] = np.nan

# print len(furniture_df['Item Code#2'].unique()) # equals 573

# ---------------------------------------------------------
# Remove the rows for which Item Code is empty
# ---------------------------------------------------------
furniture_df.dropna(inplace=True)
furniture_df.reset_index(drop=True, inplace=True)

# Check which categories need re-population in the unique values
# for category in furniture_df['Category'].unique():
#     print category + ": " + str(furniture_df.loc[furniture_df['Category'] == category, 'Category'].count())


# ---------------------------------------------------------
# Modify the df unique codes with their category, type, and material
# ---------------------------------------------------------

# Inserting values according to mappings in mappings.yaml
furniture_df['new_category'] = furniture_df['Category'].apply(lambda x: mappings['furniture'][x]['Category'])

furniture_df['new_type'] = furniture_df['Category'].apply(
    lambda x: mappings['furniture'][x]['Type'] if mappings['furniture'][x]['Type'] != 'NULL' else np.nan
)
furniture_df['new_material'] = furniture_df['Category'].apply(lambda x: mappings['furniture'][x]['Material'])

furniture_df.drop_duplicates(subset='Item Code', inplace=True)
# furniture_df.to_csv('../Data/Unique_dataset.csv', columns=['Item Code', 'new_category', 'new_type', 'new_material'])

# Modified leftover data manually and importing the modified unique entries
# NOTE --> The data is already unique and can be added to the inventory data automatically

# ---------------------------------------------------------
# Add the extra entries from the furniture sales to inventory data
# ---------------------------------------------------------
"""

modified_data = pd.read_csv('../Data/new_unique_dataset.csv', usecols=[
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

inventory_df.to_csv('../Data/furniture_sales_added_inventory.csv')
