import pandas as pd
from pathlib import Path

# Read the original CSV file
df = pd.read_csv("precipitation.csv")
df = df.drop(columns=["latitude", "longitude"], errors="ignore")
df["valid_time"] = pd.to_datetime(df["valid_time"], format="%Y-%m-%d %H:%M:%S")

# Create separate columns for Hour, Day, Month, and Year
df["Hour"] = df["valid_time"].dt.hour
df["Day"] = df["valid_time"].dt.day
df["Month"] = df["valid_time"].dt.month
df["Year"] = df["valid_time"].dt.year

# Format the precipitation data in mm
df["Total precipitation (mm)"] = df["tp"] * 1000

# Drop the original time and tp columns
df = df.drop(columns=["valid_time", "tp"])
column_order = ["Year", "Month", "Day", "Hour", "Total precipitation (mm)"]
df = df[column_order]

# Compute monthly total precipitation
monthly = df.groupby(["Year", "Month"], as_index=False)["Total precipitation (mm)"].sum()
monthly["Total precipitation (mm)"] = monthly["Total precipitation (mm)"].round(2)

# Compute the number of days with at least 1mm of precipitation
daily = df.groupby(["Year", "Month", "Day"], as_index=False)["Total precipitation (mm)"].sum()
rain_days = daily[daily["Total precipitation (mm)"] >= 1]
rain_days_count = rain_days.groupby(["Year", "Month"]).size().reset_index(name="Number of days (> 1mm of precipitation)")

# Merge monthly totals with the daily rain counts
monthly = monthly.merge(rain_days_count, on=["Year", "Month"], how="left")
monthly["Number of days (> 1mm of precipitation)"] = monthly["Number of days (> 1mm of precipitation)"].fillna(0).astype(int)

# # Save the final monthly aggregated data to CSV
# monthly.to_csv("precipitation_monthly.csv", index=False)

# Format the precipitation data in pre LSOA per year per month
data_dir = Path(__file__).resolve().parent.parent
baseline_file = data_dir / 'Base/baseline_dataset.csv'
baseline = pd.read_csv(baseline_file)  # adjust file name/path as needed
merged = baseline.merge(monthly, on=["Year", "Month"], how="left")
merged = merged.drop(columns=["LSOA code 2011", 'LSOA name 2021', "Change Indicator"], errors="ignore")

# Save the merged dataset to CSV
merged.to_csv("precipitation_finalized.csv", index=False)

print(merged.head())