import pandas as pd

# Load the police data, the lookup file and change the column names to prepare for joining
data = pd.read_csv('data/Burglary London (2010 - 2025).csv', low_memory=False)
data = data[data['Crime ID'].notna()]
lookup = pd.read_csv('data/LSOA_(2021)_to_Electoral_Ward_(2024)_to_LAD_(2024)_Best_Fit_Lookup_in_EW.csv')
lookup = lookup.rename(columns = {'LSOA21CD': 'LSOA code', 'WD24CD': 'Ward code', 'WD24NM': 'Ward name'})

# Join the data with the lookup file to get the belonging wards
df = data.join(lookup.set_index('LSOA code'), on='LSOA code', how='left')
df = df[['Crime ID', 'Month', 'Reported by', 'Falls within', 'Longitude',
       'Latitude', 'Location', 'LSOA code', 'LSOA name', 'Ward code', 'Ward name', 'Crime type',
       'Last outcome category', 'Context']]

# Get the LSOA codes which have not been connected to a ward
missing = df[df['Ward code'].isna()]['LSOA code'].unique()

# Load the LSOA 2011 to LSOA 2021 file to get the correct LSOA codes
CD11_to_CD21 = pd.read_csv('data/LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Best_Fit_Lookup_for_EW_(V2).csv')

# Create new lookup file for the missing wards
missing_21 = CD11_to_CD21[CD11_to_CD21['LSOA11CD'].isin(missing)][['LSOA11CD', 'LSOA21CD']].rename(
       columns = {'LSOA21CD': 'LSOA code'}
)
missing_lookup = missing_21.join(
       lookup.set_index('LSOA code'),
       on='LSOA code',
       how='left'
)[['LSOA11CD', 'Ward code', 'Ward name']].rename(columns = {'LSOA11CD': 'LSOA code'})

# Only update missing wards
df_missing_info = df[['LSOA code', 'Ward code', 'Ward name']].copy()

# Merge in missing wards
df_missing_info = df_missing_info.set_index('LSOA code')
filled_info = missing_lookup.set_index('LSOA code')[['Ward code', 'Ward name']]

# Fill in missing wards
df_missing_info.update(filled_info)

# Replace it in original dataframe
df['Ward code'] = df_missing_info['Ward code'].values
df['Ward name'] = df_missing_info['Ward name'].values

# Save to CSV
print('Saving to csv...')
df.to_csv('data/burglary_london.csv', index=False)
print('File saved.')

