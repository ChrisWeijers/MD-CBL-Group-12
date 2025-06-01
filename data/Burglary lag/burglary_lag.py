import pandas as pd

# Load the crimes_finalized dataset
crimes = pd.read_csv('C:/Users/20231441/OneDrive - TU Eindhoven/Documents/GitHub/MD-CBL-Group-12/data/Crimes/crimes_finalized.csv')
crimes = crimes[['LSOA code 2021', 'Year', 'Month', 'Burglary count']]

# Sort the data by LSOA code, Year and Month
crimes.sort_values(by=['LSOA code 2021', 'Year', 'Month'], inplace=True)

# Create lag columns for the past three months
crimes['Burglary count (-1 month)'] = crimes.groupby('LSOA code 2021')['Burglary count'].shift(1)
crimes['Burglary count (-2 month)'] = crimes.groupby('LSOA code 2021')['Burglary count'].shift(2)
crimes['Burglary count (-3 month)'] = crimes.groupby('LSOA code 2021')['Burglary count'].shift(3)

# Create a column for the average burglary count over the past year
crimes['Burglary (past 12 month average)'] = crimes.groupby('LSOA code 2021')['Burglary count'].transform(
    lambda x: x.shift(1).rolling(window=12, min_periods=1).mean()
)
crimes['Burglary (past 12 month average)'] = crimes['Burglary (past 12 month average)'].round(2)

# Create a column for the average burglary count over the past 6 months
crimes['Burglary (past 6 month average)'] = crimes.groupby('LSOA code 2021')['Burglary count'].transform(
    lambda x: x.shift(1).rolling(window=6, min_periods=1).mean()
)
crimes['Burglary (past 6 month average)'] = crimes['Burglary (past 6 month average)'].round(2)

# Create a column for the average burglary count over the past 3 months
crimes['Burglary (past 3 month average)'] = crimes.groupby('LSOA code 2021')['Burglary count'].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
)
crimes['Burglary (past 3 month average)'] = crimes['Burglary (past 3 month average)'].round(2)

# Clean up the data
crimes.drop(columns=['Burglary count'], inplace=True, errors='ignore')

# Save the lag dataset
crimes.to_csv('burglary_lag_finalized.csv', index=False)
print(crimes.head())