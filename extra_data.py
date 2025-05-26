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

def norm_round(cols):
    # Normalizes and rounds
    df[cols] = df[cols].div(df[cols].sum(axis=1), axis=0) * 100
    df[cols] = df[cols].round(2)
    df[cols] = df[cols].div(df[cols].sum(axis=1), axis=0) * 100
    df[cols] = df[cols].round(2)

# Load the data without the estimations
df = pd.read_csv('data/extra_data_no_estimates.csv')
df.set_index(['LSOA code', 'year', 'month'], inplace=True)

# Different categories columns
hours_cols = ['Part-time: 15 hours or less worked', 'Part-time: 16 to 30 hours worked',
                'Full-time: 31 to 48 hours worked', 'Full-time: 49 or more hours worked']
qual_cols = ['No qualifications', 'Level 1', 'Level 2', 'Apprenticeship', 'Level 3', 'Level 4+', 'Other']
imd_cols = ['imd_rank', 'imd_decile']

# Interpolate/extrapolate
print('Inter- and extrapolating data...')
all_cols = ['pop_density'] + hours_cols + qual_cols + imd_cols
df[all_cols] = df[all_cols].groupby(level=0).transform(
    lambda group: extrapolate_linear(group)
)

# Normalizing and rounding the necessary columns
norm_round(hours_cols)
norm_round(qual_cols)

# Save the data as csv
df.to_csv('data/extra_data.csv')