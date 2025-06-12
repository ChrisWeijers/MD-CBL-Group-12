import pandas as pd
import numpy as np
from pathlib import Path

# Load the burglary data
data_dir = Path(__file__).resolve().parent.parent
crimes_file = data_dir / 'Crimes/crimes_finalized.csv'
crimes = pd.read_csv(crimes_file, usecols=["LSOA code 2021", "Year", "Month", "Burglary count"])

# Make a date variable
crimes["date"] = pd.to_datetime(crimes[["Year", "Month"]].assign(DAY=1))

# Filter to the range 2011-01 through 2025-02
crimes = crimes[(crimes["date"] >= "2011-01-01") & (crimes["date"] <= "2025-04-01")]

# Pivot to get a wide dataframe
pivot = crimes.pivot_table(
    index="date",
    columns="LSOA code 2021",
    values="Burglary count",
    aggfunc="sum",
    fill_value=0
)

# Ensure that all months are present
all_months = pd.date_range(start="2011-01-01", end="2025-04-01", freq="MS")
pivot = pivot.reindex(all_months, fill_value=0)

# Shift down by one month so that “lag1” for 2023-06 reflects May 2023
shifted = pivot.shift(1).fillna(0)

# Compute rolling standard deviations on the shifted data

std3_df = shifted.rolling(window=3, min_periods=1).std(ddof=0).stack().reset_index().rename(columns={0: "Standard deviation (3 months)", "level_1": "LSOA code 2021", "level_0": "date"})
std6_df = shifted.rolling(window=6, min_periods=1).std(ddof=0).stack().reset_index().rename(columns={0: "Standard deviation (6 months)", "level_1": "LSOA code 2021", "level_0": "date"})
std12_df = shifted.rolling(window=12, min_periods=1).std(ddof=0).stack().reset_index().rename(columns={0: "Standard deviation (12 months)", "level_1": "LSOA code 2021", "level_0": "date"})

# Extract Year & Month from the “date” column for merging
for df in [std3_df, std6_df, std12_df]:
    df["Year"] = df["date"].dt.year
    df["Month"] = df["date"].dt.month

# Load your baseline dataframe
baseline_file = data_dir / 'Base/baseline_dataset.csv'
baseline = pd.read_csv(
    baseline_file, dtype={"LSOA code 2021": str, "Year": int, "Month": int})
baseline = baseline.drop_duplicates(subset=["LSOA code 2021", "Year", "Month"])

# 8) Merge stds into the baseline
data_with_stds = baseline.merge(
    std3_df[["LSOA code 2021", "Year", "Month", "Standard deviation (3 months)"]],
    on=["LSOA code 2021", "Year", "Month"],
    how="left"
).merge(
    std6_df[["LSOA code 2021", "Year", "Month", "Standard deviation (6 months)"]],
    on=["LSOA code 2021", "Year", "Month"],
    how="left"
).merge(
    std12_df[["LSOA code 2021", "Year", "Month", "Standard deviation (12 months)"]],
    on=["LSOA code 2021", "Year", "Month"],
    how="left"
)

# Clean up the dataframe
data_with_stds = data_with_stds.drop(columns=["LSOA code 2011", "LSOA name 2021", "Change Indicator"])

# Save the dataframe to csv
data_with_stds.to_csv("rolling_std_finalized.csv", index=False)
print(data_with_stds.head(20))