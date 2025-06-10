from astral import LocationInfo
from astral.sun import sun
import datetime
import pandas as pd
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent
baseline_file = data_dir / 'Base/baseline_dataset.csv'

# 1. Set up the London location
london = LocationInfo("London", "England", "Europe/London", 51.503579, -0.131917)

# 2. Prepare a DataFrame indexed by each month from Jan 2011 to May 2026
months = pd.period_range("2011-01", "2026-05", freq="M")
df = pd.DataFrame(index=months, columns=["daylight_minutes"])
df["daylight_minutes"] = 0.0

# 3. Loop through each day, compute sunrise/sunset, accumulate by month
for year in range(2011, 2027):
    if year == 2026:
        month_range = range(1, 6)
    else:
        month_range = range(1, 13)
    for month in month_range:
        days_in_month = (pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd()).day
        for day in range(1, days_in_month + 1):
            date = datetime.date(year, month, day)
            s = sun(london.observer, date=date, tzinfo=london.timezone)
            minutes = round((s["sunset"] - s["sunrise"]).total_seconds() / 60, 0)
            df.loc[pd.Period(f"{year}-{month:02d}", freq="M"), "daylight_minutes"] += minutes

# 4. Reset index and split into Year and Month
df = df.reset_index().rename(columns={"index": "year_month"})
df["Year"] = df["year_month"].dt.year
df["Month"] = df["year_month"].dt.month

# 5. Clean up the dataframe
df["daylight_minutes"] = df["daylight_minutes"].astype(int)
df = df.drop(columns=["year_month"])
df = df.rename(columns={"daylight_minutes": "Daylight minutes"})
df = df[["Year", "Month", "Daylight minutes"]]

# Convert daylight minutes into daylight hours, round to 0 decimal points, and update column name
df["Daylight hours"] = (df["Daylight minutes"] / 60).round(0).astype(int)
df = df.drop(columns=["Daylight minutes"])
df = df[["Year", "Month", "Daylight hours"]]

# Format the daylight data per LSOA per year per month
baseline = pd.read_csv(baseline_file,
                        dtype={"LSOA code 2021": str})
merged = baseline.merge(df, on=["Year", "Month"], how="left")
merged = merged.drop(columns=["LSOA code 2011", "LSOA name 2021", "Change Indicator"], errors="ignore")

# Save the merged dataset to CSV
merged.to_csv("daylight_finalized.csv", index=False)
print(merged.head())