import pandas as pd
from itertools import product

# 1. Load and reshape pop density 2011 until 2022 -------------------------------------------
df_pop = (
    pd.read_excel(
        'data/populationdensity20112022.xlsx',
        sheet_name='Mid-2011 to mid-2022 LSOA 2021',
        usecols=['LSOA 2021 Code',
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
       'Mid-2022: People per Sq Km']
    )
    .rename(columns={
    'LSOA 2021 Code': 'LSOA code',
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
)

# Melt population density into long form for years 2011 and 2021
df_pop_long = df_pop.melt(
    id_vars=['LSOA code'],
    value_vars=['2011_pop_density',
    '2012_pop_density',
    '2013_pop_density',
    '2014_pop_density',
    '2015_pop_density',
    '2016_pop_density',
    '2017_pop_density',
    '2018_pop_density',
    '2019_pop_density',
    '2020_pop_density',
    '2021_pop_density',
    '2022_pop_density'],
    var_name='year_pop',
    value_name='pop_density'
)
# Extract year from column name
df_pop_long['year'] = df_pop_long['year_pop'].str.extract(r"(\d{4})").astype(int)
df_pop_long = df_pop_long[['LSOA code', 'year', 'pop_density']]
# Set MultiIndex; note these rows only have yearly data so we assign month=1
df_pop_long['month'] = 1

# 2. Create full flat grid DataFrame -----------------------------------------------------
lsoas  = df_pop['LSOA code'].unique()
years  = range(2011, 2026)
months = range(1, 13)
full_grid = pd.DataFrame(
    list(product(lsoas, years, months)),
    columns=['LSOA code', 'year', 'month']
)

df = full_grid.copy()

# 3. Merge population density onto every row where year matches -------------------------
df = df.merge(
    df_pop_long,
    on=['LSOA code', 'year', 'month'],
    how='left'
)

# 4. Load & prepare hours worked for 2011 & 2021 -----------------------------------------
def load_hours(path, year):
    dfh = pd.read_csv(path)
    drop_cols = [c for c in dfh.columns if 'super output area' in c.lower()]
    dfh = dfh.drop(columns=drop_cols, errors='ignore')
    dfh = dfh.rename(columns={'mnemonic': 'LSOA code'})
    dfh['year'] = year
    dfh['month'] = 1
    return dfh

dfh_2011 = load_hours('data/Hours worked/hours_worked_2011.csv', 2011)
dfh_2021 = load_hours('data/Hours worked/hours_worked_2021.csv', 2021)
df_hours = pd.concat([dfh_2011, dfh_2021], ignore_index=True)

# Melt and pivot hours, then merge onto df
hours_long = df_hours.melt(
    id_vars=['LSOA code', 'year', 'month'],
    var_name='hours_cat',
    value_name='pct_hours'
)
df_hours_pivot = hours_long.pivot_table(
    index=['LSOA code', 'year', 'month'],
    columns='hours_cat',
    values='pct_hours'
).reset_index()

df = df.merge(
    df_hours_pivot,
    on=['LSOA code', 'year', 'month'],
    how='left'
)

# 5. Load & prepare qualifications for 2011 & 2021 --------------------------------------
def load_quals(path, year, drop_cols, rename_map):
    dfq = pd.read_csv(path)
    dfq = dfq.drop(columns=drop_cols, errors='ignore')
    dfq = dfq.rename(columns=rename_map)
    dfq['year'] = year
    dfq['month'] = 1
    return dfq

# Qualification mappings
drop11 = ['2011 super output area - lower layer', 'All categories: Highest level of qualification']
map11 = {
    'mnemonic': 'LSOA code',
    'Highest level of qualification: Level 1 qualifications': 'Level 1',
    'Highest level of qualification: Level 2 qualifications': 'Level 2',
    'Highest level of qualification: Apprenticeship': 'Apprenticeship',
    'Highest level of qualification: Level 3 qualifications': 'Level 3',
    'Highest level of qualification: Level 4 qualifications and above': 'Level 4+',
    'Highest level of qualification: Other qualifications': 'Other',
}
drop21 = ['2021 super output area - lower layer', 'Total: All usual residents aged 16 years and over']
map21 = {
    'mnemonic': 'LSOA code',
    'Level 1 and entry level qualifications': 'Level 1',
    'Level 2 qualifications': 'Level 2',
    'Level 3 qualifications': 'Level 3',
    'Level 4 qualifications or above': 'Level 4+',
    'Other qualifications': 'Other',
}

dfq_2011 = load_quals('data/Education/qualifications_2011.csv', 2011, drop11, map11)
dfq_2021 = load_quals('data/Education/qualifications_2021.csv', 2021, drop21, map21)
df_quals = pd.concat([dfq_2011, dfq_2021], ignore_index=True)

# Melt and pivot quals, then merge onto df
quals_long = df_quals.melt(
    id_vars=['LSOA code', 'year', 'month'],
    var_name='qual_level',
    value_name='pct_qual'
)
df_quals_pivot = quals_long.pivot_table(
    index=['LSOA code', 'year', 'month'],
    columns='qual_level',
    values='pct_qual'
).reset_index()

df = df.merge(
    df_quals_pivot,
    on=['LSOA code', 'year', 'month'],
    how='left'
)

# 6. Load & prepare IMD for 2015 & 2019
df_imd_2015 = (
    pd.read_excel(
                'data/File_1_ID_2015_Index_of_Multiple_Deprivation.xlsx',
                sheet_name='IMD 2015',
                usecols=['LSOA code (2011)',
                        'Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)',
                        'Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)',
                        ]
                ).rename(columns={'LSOA code (2011)': 'LSOA code',
                                  'Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)': 'imd_rank',
                                  'Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)': 'imd_decile'})
               )
df_imd_2019 = (
    pd.read_excel(
                'data/File_1_IMD2019_Index_of_Multiple_Deprivation.xlsx',
                sheet_name='IMD2019',
                usecols=['LSOA code (2011)',
                        'Index of Multiple Deprivation (IMD) Rank',
                        'Index of Multiple Deprivation (IMD) Decile',
                        ]
                ).rename(columns={'LSOA code (2011)': 'LSOA code',
                                  'Index of Multiple Deprivation (IMD) Rank': 'imd_rank',
                                  'Index of Multiple Deprivation (IMD) Decile': 'imd_decile'
                                  })
               )
df_imd_2015['year'] = 2015
df_imd_2015['month'] = 1
df_imd_2019['year'] = 2019
df_imd_2019['month'] = 1

df_imd = pd.concat([df_imd_2015, df_imd_2019], ignore_index=True)

imd_long = df_imd.melt(
    id_vars=['LSOA code', 'year', 'month'],
    var_name='imd',
    value_name='imd_score'
)
df_imd_pivot = imd_long.pivot_table(
    index=['LSOA code', 'year', 'month'],
    columns='imd',
    values='imd_score'
).reset_index()

df = df.merge(
    df_imd_pivot,
    on=['LSOA code', 'year', 'month'],
    how='left'
)

# 7. Load London daylight hours
df_daylight = (pd.read_excel('data/london_daylight.xlsx',
                            usecols=['Month', 'Hours of daylight']).rename(columns={'Month': 'month'}))

df_daylight['Hours of daylight'] = df_daylight['Hours of daylight'].str.replace(' h', '').apply(
    lambda x: int(x.split(':')[0]) + int(x.split(':')[1]) / 60
).round(2)

df = df.merge(
    df_daylight,
    on=['month'],
    how='left'
)

# 7. Finalize and save ------------------------------------------------------------------
df = df.sort_values(['LSOA code', 'year', 'month'])
df.to_csv('data/extra_data_no_estimates.csv', index=False)
print(df.head())