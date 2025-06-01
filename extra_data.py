import pandas as pd
import numpy as np

def flatten_value(x):
    while isinstance(x, tuple):
        x = x[0]
    return x

def extrapolate_linear(group):
    # Flatten all values in the series
    group = group.apply(flatten_value)
    known = group.dropna()
    if len(known) < 2:
        return group.interpolate(method='linear', limit_direction='both')
    # For monthly estimation, compute a "time" value as: time = year + (month-1)/12
    # The index is assumed to be (LSOA code, year, month)
    times = group.index.map(lambda idx: idx[1] + (idx[2]-1)/12)
    known_times = group.dropna().index.map(lambda idx: idx[1] + (idx[2]-1)/12)
    t_start = known_times.min()
    t_end = known_times.max()
    # Find the corresponding start and end values
    v_start = flatten_value(known[known_times == t_start].iloc[0])
    v_end = flatten_value(known[known_times == t_end].iloc[0])
    slope = (v_end - v_start) / (t_end - t_start)
    filled = group.copy()
    for idx in group.index:
        t_val = idx[1] + (idx[2]-1)/12
        if pd.isna(filled.loc[idx]):
            filled.loc[idx] = v_start + slope * (t_val - t_start)
    return filled

# Load the population density dataset
df_pop_density = pd.read_excel('C:/Users/20231441/OneDrive - TU Eindhoven/Documents/GitHub/MD-CBL-Group-12/data/Population density/populationdensity20112022.xlsx',
                               sheet_name='Mid-2011 to mid-2022 LSOA 2021')
df_pop_density = df_pop_density[['LSOA 2021 Code', 'LSOA 2021 Name',
       'Mid-2011: People per Sq Km',
       'Mid-2012: People per Sq Km',
       'Mid-2013: People per Sq Km',
       'Mid-2014: People per Sq Km',
       'Mid-2015: People per Sq Km',
       'Mid-2016: People per Sq Km',
       'Mid-2017: People per Sq Km',
       'Mid-2018: People per Sq Km',
       'Mid-2019: People per Sq Km',
       'Mid-2020: People per Sq Km',
       'Mid-2021: People per Sq Km',
       'Mid-2022: People per Sq Km']].rename(columns={
    'LSOA 2021 Code': 'LSOA code',
    'LSOA 2021 Name': 'LSOA name',
    'Mid-2011: People per Sq Km': '2011_pop_density',
    'Mid-2012: People per Sq Km': '2012_pop_density',
    'Mid-2013: People per Sq Km': '2013_pop_density',
    'Mid-2014: People per Sq Km': '2014_pop_density',
    'Mid-2015: People per Sq Km': '2015_pop_density',
    'Mid-2016: People per Sq Km': '2016_pop_density',
    'Mid-2017: People per Sq Km': '2017_pop_density',
    'Mid-2018: People per Sq Km': '2018_pop_density',
    'Mid-2019: People per Sq Km': '2019_pop_density',
    'Mid-2020: People per Sq Km': '2020_pop_density',
    'Mid-2021: People per Sq Km': '2021_pop_density',
    'Mid-2022: People per Sq Km': '2022_pop_density',
})

# Create the MultiIndex for monthly data
lsoa_codes = list(df_pop_density['LSOA code'])
years = list(range(2011, 2026))
months = list(range(1, 13))
iterables = [lsoa_codes, years, months]
index = pd.MultiIndex.from_product(iterables, names=["LSOA code", "year", "month"])

# Reshape population density data from wide to long format
df_pop_long = df_pop_density.melt(
    id_vars=['LSOA code', 'LSOA name'],
    value_vars=[col for col in df_pop_density.columns if 'pop_density' in col],
    var_name='year_pop',
    value_name='pop_density'
)
# Convert the 'year_pop' strings to int (extract year)
df_pop_long['year'] = df_pop_long['year_pop'].str.extract(r'(\d{4})').astype(int)
df_pop_long.drop(columns='year_pop', inplace=True)
# Set MultiIndex; note these rows only have yearly data so we assign month=1
df_pop_long['month'] = 1
df_pop_long.set_index(['LSOA code', 'year', 'month'], inplace=True)

# Create the full DataFrame; missing months (and even years) will be estimated.
print('Adding population density...')
df = pd.DataFrame(df_pop_long['pop_density'], index=index)
# Interpolate/extrapolate population density
print('Inter- and extrapolating pop_density...')
df['pop_density'] = df.groupby(level=0)['pop_density'].transform(
    lambda group: extrapolate_linear(group)
)
df['pop_density'] = df['pop_density'].round(2)

# Load the hours worked datasets
df_hours_2011 = pd.read_csv('C:/Users/20231441/OneDrive - TU Eindhoven/Documents/GitHub/MD-CBL-Group-12/data/Hours worked/hours_worked_2011.csv')
df_hours_2021 = pd.read_csv('data/Hours worked/hours_worked_2021.csv')
df_hours_2011 = df_hours_2011.drop(columns='2011 super output area - lower layer').rename(columns={
    'mnemonic': 'LSOA code',
})
df_hours_2021 = df_hours_2021.drop(columns='2021 super output area - lower layer').rename(columns={
    'mnemonic': 'LSOA code',
})

# Load the qualifications datasets
df_qual_2011 = pd.read_csv('data/Education/qualifications_2011.csv')
df_qual_2021 = pd.read_csv('data/Education/qualifications_2021.csv')
df_qual_2011 = df_qual_2011.drop(columns=[
    '2011 super output area - lower layer', 'All categories: Highest level of qualification'
]).rename(columns={
    'mnemonic': 'LSOA code',
    'Highest level of qualification: Level 1 qualifications': 'Level 1',
    'Highest level of qualification: Level 2 qualifications': 'Level 2',
    'Highest level of qualification: Apprenticeship': 'Apprenticeship',
    'Highest level of qualification: Level 3 qualifications': 'Level 3',
    'Highest level of qualification: Level 4 qualifications and above': 'Level 4+',
    'Highest level of qualification: Other qualifications': 'Other',
})
df_qual_2021 = df_qual_2021.drop(columns=[
    '2021 super output area - lower layer', 'Total: All usual residents aged 16 years and over'
]).rename(columns={
    'mnemonic': 'LSOA code',
    'Level 1 and entry level qualifications': 'Level 1',
    'Level 2 qualifications': 'Level 2',
    'Level 3 qualifications': 'Level 3',
    'Level 4 qualifications or above': 'Level 4+',
    'Other qualifications': 'Other',
})

# Add year columns
df_hours_2011['year'] = 2011
df_hours_2021['year'] = 2021
df_qual_2011['year'] = 2011
df_qual_2021['year'] = 2021

# For the purposes of monthly interpolation, assign month=1 to the yearly data
df_hours_2011['month'] = 1
df_hours_2021['month'] = 1
df_qual_2011['month'] = 1
df_qual_2021['month'] = 1

# Combine the separate year datasets
df_hours = pd.concat([df_hours_2011, df_hours_2021], ignore_index=True)
df_qual = pd.concat([df_qual_2011, df_qual_2021], ignore_index=True)

# Reshape hours worked and qualifications from wide to long format
df_hours_long = df_hours.melt(
    id_vars=['LSOA code', 'year', 'month'],
    var_name='hours_worked',
    value_name='percentage'
)
df_qual_long = df_qual.melt(
    id_vars=['LSOA code', 'year', 'month'],
    var_name='qualification',
    value_name='percentage'
)

# Ensure meaningful order for later
df_hours_long['hours_worked'] = pd.Categorical(
    df_hours_long['hours_worked'],
    categories=['Part-time: 15 hours or less worked', 'Part-time: 16 to 30 hours worked',
                'Full-time: 31 to 48 hours worked', 'Full-time: 49 or more hours worked'],
    ordered=True
)
df_qual_long['qualification'] = pd.Categorical(
    df_qual_long['qualification'],
    categories=['No qualifications', 'Level 1', 'Level 2', 'Apprenticeship', 'Level 3', 'Level 4+', 'Other'],
    ordered=True
)

# Pivot to get hours worked and qualification levels as columns again
df_hours_pivot = df_hours_long.pivot_table(index=['LSOA code', 'year', 'month'],
                                   columns='hours_worked',
                                   values='percentage',
                                   observed=False)
df_qual_pivot = df_qual_long.pivot_table(index=['LSOA code', 'year', 'month'],
                                   columns='qualification',
                                   values='percentage',
                                   observed=False)

# Add the data to the MultiIndex dataframe for interpolating
hours_cols = df_hours_pivot.columns
for col in hours_cols:
    df[col] = df_hours_pivot[col]
qual_cols = df_qual_pivot.columns
for col in qual_cols:
    df[col] = df_qual_pivot[col]

def convert_all_nan_list_to_nan(lst):
    return np.nan if all(pd.isna(x) for x in lst) else lst

df_standard = df.copy()

# Convert hours worked
df_standard['hours_worked_percentages'] = df[hours_cols].values.tolist()
df_standard['hours_worked_percentages'] = df_standard['hours_worked_percentages'].apply(convert_all_nan_list_to_nan)

# Convert qualification
df_standard['qualification_percentages'] = df[qual_cols].values.tolist()
df_standard['qualification_percentages'] = df_standard['qualification_percentages'].apply(convert_all_nan_list_to_nan)

# Drop original columns
df_standard = df_standard.drop(columns=hours_cols)
df_standard = df_standard.drop(columns = qual_cols)
# df_standard.to_csv('no_estimation_data.csv')
# Make sure the df is sorted
df = df.sort_index()

# Interpolate/extrapolate, normalize and round for hours worked
print('Inter- and extrapolating hours worked...')
df[hours_cols] = df[hours_cols].groupby(level=0).transform(
    lambda group: extrapolate_linear(group)
)
df[hours_cols] = df[hours_cols].div(df[hours_cols].sum(axis=1), axis=0) * 100
df[hours_cols] = df[hours_cols].round(2)
df[hours_cols] = df[hours_cols].div(df[hours_cols].sum(axis=1), axis=0) * 100
df[hours_cols] = df[hours_cols].round(2)

print('Processing hours worked percentages...')
# Keep the original hours worked columns (each as a separate column)

# Interpolate/extrapolate, normalize and round for qualifications
print('Inter- and extrapolating qualifications...')
df[qual_cols] = df[qual_cols].groupby(level=0).transform(
    lambda group: extrapolate_linear(group)
)
df[qual_cols] = df[qual_cols].div(df[qual_cols].sum(axis=1), axis=0) * 100
df[qual_cols] = df[qual_cols].round(2)
df[qual_cols] = df[qual_cols].div(df[qual_cols].sum(axis=1), axis=0) * 100
df[qual_cols] = df[qual_cols].round(2)

df.to_csv('extra_data.csv')