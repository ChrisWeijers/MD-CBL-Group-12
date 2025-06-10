import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

# Load the population density data
density_raw = pd.read_excel("populationdensity20112022.xlsx", sheet_name="Mid-2011 to mid-2022 LSOA 2021")
density_raw.rename(columns={"LSOA 2021 Code": "LSOA code 2021"}, inplace=True)

# For each year 2011 to 2022, extract the density values and assign Month = 6
density_list = []
for yr in range(2011, 2023):
    col_name = f"Mid-{yr}: People per Sq Km"
    if col_name in density_raw.columns:
        tmp = density_raw[["LSOA code 2021", col_name]].copy()
        tmp["Year"] = yr
        tmp["Month"] = 6  # using mid-year as month 6
        tmp.rename(columns={col_name: "People per Km^2"}, inplace=True)
        density_list.append(np.round(tmp,2))
    else:
        print(f"Column '{col_name}' not found in file, skipping.")
density_all = pd.concat(density_list, ignore_index=True)

# Merge with the baseline dataset that contains every LSOA, Year & Month combination
data_dir = Path(__file__).resolve().parent.parent
baseline_file = data_dir / 'Base/baseline_dataset.csv'
baseline = pd.read_csv(baseline_file)
density_combined = baseline.merge(density_all, on=["LSOA code 2021", "Year", "Month"], how="left")

# Create a continuous time variable
density_combined["time"] = density_combined["Year"] * 12 + density_combined["Month"]

# Estimate density values for months where data is missing
density_var = "People per Km^2"

def interpolate_density(group):
    group = group.sort_values("time")
    mask = group[density_var].notna()
    if mask.sum() >= 2:
        x_known = group.loc[mask, "time"]
        y_known = group.loc[mask, density_var]
        f = interp1d(x_known, y_known, kind="linear", fill_value="extrapolate", bounds_error=False)
        group[density_var] = f(group["time"])
    return np.round(group, 2)

density_interp = density_combined.groupby("LSOA code 2021").apply(interpolate_density)

# Clean up and finalize
density_interp.drop(columns=['LSOA code 2011', 'LSOA name 2021', 'Change Indicator', 'time'], inplace=True)

# Save the finalized population density data
density_interp.to_csv("population_density_finalized.csv", index=False)
print(density_interp.head())