import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
import re
from functools import reduce

# Create function for loading the household income data.
def load_household_income(file_path, year, weekly=False):
    income_list = []
    if not weekly:
        sheet_names = ['Total annual income', 'Net annual income', 'Net income before housing costs',
                       'Net income after housing costs']
    else:
        sheet_names = ['Total weekly income', 'Net weekly income', 'Net income before housing costs',
                       'Net income after housing costs']
    for sheet in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet,
                            usecols=lambda col: col == "MSOA code" or bool(re.search("income", col, flags=re.IGNORECASE)))
        income_list.append(df)
    df = reduce(lambda  left,right: pd.merge(left,right,on=['MSOA code'],
                                            how='outer'), income_list)
    df['Year'] = year
    df['Month'] = 4
    if weekly:
        columns_old = ['Total weekly income (£)', 'Net weekly income (£)', 'Net income before housing costs (£)',
                   'Net income after housing costs (£)']
        columns_new = ['Total annual income (£)', 'Net annual income (£)', 'Net annual income before housing costs (£)',
                   'Net annual income after housing costs (£)']
        df[columns_new] = df[columns_old] * 52
        df.drop(columns=columns_old, inplace=True)
    return df


# Load the household income data for the years 2012, 2014, 2016, 2018 and 2020.
hi_2012 = load_household_income('1smallareaincomeestimatesdata201112.xlsx', 2012, weekly=True)
hi_2014 = load_household_income('1smallareaincomeestimatesdata2014.xlsx', 2014, weekly=True)
hi_2016 = load_household_income('1smallareaincomeestimatesdata2016.xls', 2016)
hi_2018 = load_household_income('incomeestimatesforsmallareasdatasetfinancialyearending20181.xlsx', 2018)
hi_2020 = load_household_income('annualincome2020.xlsx', 2020)

# Combine the dataframes.
df_hi = pd.concat([hi_2012, hi_2014, hi_2016, hi_2018, hi_2020])

# Load the LSOA to MSOA lookup file.
msoa_lsoa = pd.read_csv('OAs_to_LSOAs_to_MSOAs_to_LEP_to_LAD_(April_2023)_Lookup_in_England.csv',
                        usecols=['LSOA21CD', 'MSOA21CD'])
msoa_lsoa = msoa_lsoa.drop_duplicates().groupby('MSOA21CD', group_keys=True)[['LSOA21CD']].apply(lambda x: x)

# Load the MSOA 2011 to MSOA 2021 lookup file.
all_msoas = pd.read_csv(
    'MSOA_(2011)_to_MSOA_(2021)_to_Local_Authority_District_(2022)_Lookup_for_England_and_Wales_-5379446518771769392.csv')
all_msoas = all_msoas.drop(columns=["LAD22NMW", "ObjectId"], errors="ignore")

# Filter to only Greater London area LSOAs.
london_lads = {
    "Camden", "Greenwich", "Hackney", "Hammersmith and Fulham", "Islington",
    "Kensington and Chelsea", "Lambeth", "Lewisham", "Southwark", "Tower Hamlets",
    "Wandsworth", "Westminster", "Barking and Dagenham", "Barnet", "Bexley",
    "Brent", "Bromley", "Croydon", "Ealing", "Enfield", "Haringey", "Harrow",
    "Havering", "Hillingdon", "Hounslow", "Kingston upon Thames", "Merton",
    "Newham", "Redbridge", "Richmond upon Thames", "Sutton", "Waltham Forest",
}
london_msoas = all_msoas[all_msoas["LAD22NM"].isin(london_lads)].copy()

print(f"Found {len(london_msoas)} MSOAs in Greater London")

# Merge the household income data with the MSOA 2011 to MSOA 2021 lookup file to get the 2021 MSOA code.
hi_mapped = pd.merge(df_hi, london_msoas, left_on='MSOA code', right_on='MSOA11CD', how='left')

# Split the data into two groups
# Rows with indicator "U - Unchanged", "S - Split" or "X - Other changes" (which in this case is just one split)
data_USX = hi_mapped[hi_mapped['CHGIND'].isin(['U', 'S', 'X'])]
data_USX_reduced = data_USX[['MSOA21CD', 'Year', 'Month', 'Total annual income (£)',
                             'Net annual income (£)', 'Net annual income before housing costs (£)',
                   'Net annual income after housing costs (£)']].copy()

# Rows with indicator "M - Merged"
data_M = hi_mapped[hi_mapped['CHGIND'] == 'M']
data_M_avg = (
    data_M.groupby(['MSOA21CD', 'Year', 'Month'])[['Total annual income (£)', 'Net annual income (£)',
    'Net annual income before housing costs (£)', 'Net annual income after housing costs (£)']]
    .mean()
    .reset_index()
)

# Combine the two groups
combined_income = pd.concat([data_USX_reduced, data_M_avg], ignore_index=True)

# Merge the combined income data with the MSOA 2021 to LSOA 2021 lookup file to get the 2021 LSOA code.
df_lsoa_income = combined_income.merge(msoa_lsoa, on='MSOA21CD', how='left').rename(
    columns={'LSOA21CD': 'LSOA code 2021'})
df_lsoa_income.drop(columns=['MSOA21CD'], inplace=True)
df_lsoa_income.sort_values(by=['LSOA code 2021', 'Year', 'Month'], inplace=True)

# Load the baseline dataset and merge with the created income dataframe.
data_dir = Path(__file__).resolve().parent.parent
baseline_file = data_dir / 'Base/baseline_dataset.csv'
baseline = pd.read_csv(baseline_file,
                       usecols=['LSOA code 2021', 'Year', 'Month'])
income = baseline.merge(df_lsoa_income, on=['LSOA code 2021', 'Year', 'Month'], how='left').drop_duplicates()

# Estimate the 'Total annual income (£)' for the missing months
income['time'] = income['Year'] * 12 + income['Month']
income_vars = ['Total annual income (£)', 'Net annual income (£)',
    'Net annual income before housing costs (£)', 'Net annual income after housing costs (£)']


def interpolate_extrapolate(group):
    group = group.sort_values("time")
    for col in income_vars:
        # Get available (non-missing) pairs
        mask = group[col].notna()
        if mask.sum() >= 2:
            x_known = group.loc[mask, "time"]
            y_known = group.loc[mask, col]
            # f will perform linear interpolation and extrapolation as needed.
            f = interp1d(x_known, y_known, kind="linear", fill_value="extrapolate", bounds_error=False)
            group[col] = f(group["time"])
    return group


# Apply the interpolation.
income_interp = income.groupby(by='LSOA code 2021', group_keys=False).apply(interpolate_extrapolate)
income_interp.drop(columns=['time'], inplace=True)

# Clean up the dataframe
income_interp = income_interp.rename(columns={'Total annual income (£)': 'Total annual income (GBP)',
    'Net annual income (£)': 'Net annual income (GBP)',
    'Net annual income before housing costs (£)': 'Net income before housing costs (GBP)',
    'Net annual income after housing costs (£)': 'Net income after housing costs (GBP)'
                                              })

# Save the estimated dataset to CSV.
income_interp.to_csv('household_income_finalized.csv', index=False)
print(income_interp.info())
