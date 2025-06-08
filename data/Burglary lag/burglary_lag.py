import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent
crimes_file = data_dir / 'Crimes/crimes_finalized.csv'

# Load the crimes_finalized dataset
crimes = pd.read_csv(crimes_file)
crimes = crimes[['LSOA code 2021', 'Year', 'Month', 'Burglary count']]

# Sort the data by LSOA code, Year and Month
crimes.sort_values(by=['LSOA code 2021', 'Year', 'Month'], inplace=True)

# Create lag columns for the past three months
crimes['Burglary count (-1 month)'] = crimes.groupby('LSOA code 2021')['Burglary count'].shift(1)
crimes['Burglary count (-2 month)'] = crimes.groupby('LSOA code 2021')['Burglary count'].shift(2)
crimes['Burglary count (-3 month)'] = crimes.groupby('LSOA code 2021')['Burglary count'].shift(3)

# Create a column for the average burglary count over the past year
crimes['Burglary (12 month EMA)'] = crimes['Burglary count'].shift(1).ewm(span=12, min_periods=1, adjust=False).mean().round(2)

# Create a column for the average burglary count over the past 6 months
crimes['Burglary (6 month EMA)'] = crimes['Burglary count'].shift(1).ewm(span=6, min_periods=1, adjust=False).mean().round(2)

# Create a column for the average burglary count over the past 3 months
crimes['Burglary (3 month EMA)'] = crimes['Burglary count'].shift(1).ewm(span=3, min_periods=1, adjust=False).mean().round(2)

# Clean up the data
crimes.drop(columns=['Burglary count'], inplace=True, errors='ignore')
crimes.reset_index(drop=True, inplace=True)

# Save the lag dataset
crimes.to_csv('burglary_lag_finalized.csv', index=False)
print(crimes.head())