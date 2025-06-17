import pandas as pd
from itertools import product
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent
lsoa_file = data_dir / "LSOA_changes/london_lsoa11_lsoa21_lad22_ward24.csv"

# 1. Load the LSOA dataset
lsoa_df = pd.read_csv(lsoa_file)
lsoa_df = lsoa_df.drop(columns=["LSOA11NM", "LAD22CD", "LAD22NM", "WD24CD", "WD24NM"], errors="ignore")
lsoa_df = lsoa_df.rename(columns={"LSOA11CD": "LSOA code 2011"})
lsoa_df = lsoa_df.rename(columns={"LSOA21CD": "LSOA code 2021"})
lsoa_df = lsoa_df.rename(columns={"LSOA21NM": "LSOA name 2021"})
lsoa_df = lsoa_df.rename(columns={"CHGIND": "Change Indicator"})

# 2. Build a Year-Month grid:
years_full = list(range(2011, 2026))
months_full = list(range(1, 13))
grid_full = pd.DataFrame(list(product(years_full, months_full)), columns=["Year", "Month"])

years_partial = [2026]
months_partial = list(range(1, 5))
grid_partial = pd.DataFrame(list(product(years_partial, months_partial)), columns=["Year", "Month"])

year_month = pd.concat([grid_full, grid_partial], ignore_index=True)

# 3. Cross join the lsoa_df with the Year-Month grid.
baseline = lsoa_df.merge(year_month, how="cross")
baseline = baseline.sort_values(["LSOA code 2021", "LSOA name 2021", "Year", "Month"]).reset_index(drop=True)

# 4. Save the baseline dataset to CSV
baseline.to_csv(data_dir / "Base/baseline_dataset.csv", index=False)
print("Baseline dataset created with", baseline.shape[0], "rows")
print(baseline.head())