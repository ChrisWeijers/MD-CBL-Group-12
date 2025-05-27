import pandas as pd

# Load the aggregated data
data_file = 'All Crimes London Aggregated (2010 - 2025).csv'
data = pd.read_csv(data_file)

# Pivot the data to make each crime type a column
pivoted_data = data.pivot_table(
    index=['Year', 'Month', 'LSOA code'],
    columns='Crime type',
    values='Incident Count',
    fill_value=0
).reset_index()

# Rename crime columns for clarity
pivoted_data.columns.name = None  # Remove the name of the columns index
pivoted_data = pivoted_data.rename_axis(None, axis=1)

# Load the third CSV that I'm using only to get all of the LSOA codes
base_data = 'C:/Users/20231441/OneDrive - TU Eindhoven/Documents/GitHub/MD-CBL-Group-12/data/Base/baseline_dataset.csv'
base_data = pd.read_csv(base_data, header=2)
base_data = base_data[['Codes']]
base_data.rename(columns={'Codes': 'LSOA code'}, inplace=True)

# Make a dataframe "grid" with every year, month, and LSOA code combination
unique_year_month = data[['Year', 'Month']].drop_duplicates()
unique_lsoa = pd.DataFrame(base_data['LSOA code'].unique(), columns=['LSOA code'])
grid = unique_year_month.merge(unique_lsoa, how='cross')

# Merge the grid with the actual burglary data
full_data = pd.merge(grid, pivoted_data, on=['Year','Month','LSOA code'], how='left')
full_data.fillna(0, inplace=True)

# Encode categorical variable for LSOA code while preserving the original
full_data['LSOA code encoded'] = full_data['LSOA code'].astype('category').cat.codes