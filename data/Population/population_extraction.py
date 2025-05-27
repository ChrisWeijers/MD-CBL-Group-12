import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# 1. Load Population Data from Excel sheets
pop_files = [
    "sapelsoasyoa20112014.xlsx",
    "sapelsoasyoa20152018.xlsx",
    "sapelsoasyoa20192022.xlsx"
]

sheet_names_list = [
    ["Mid-2011 LSOA 2021", "Mid-2012 LSOA 2021", "Mid-2013 LSOA 2021", "Mid-2014 LSOA 2021"],
    ["Mid-2015 LSOA 2021", "Mid-2016 LSOA 2021", "Mid-2017 LSOA 2021", "Mid-2018 LSOA 2021"],
    ["Mid-2019 LSOA 2021", "Mid-2020 LSOA 2021", "Mid-2021 LSOA 2021", "Mid-2022 LSOA 2021"]
]

pop_list = []
for file, sheet_names in zip(pop_files, sheet_names_list):
    xls = pd.ExcelFile(file)
    for sheet in sheet_names:
        if sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, header=3)
            try:
                year_str = sheet.split("-")[1].split(" ")[0]
                df["Year"] = int(year_str)
            except Exception as e:
                print(f"Could not extract year from sheet '{sheet}': {e}")
                df["Year"] = np.nan
            pop_list.append(df)
        else:
            print(f"Sheet '{sheet}' not found in file '{file}', skipping.")

pop_all = pd.concat(pop_list, ignore_index=True)
pop_all.rename(columns={"LSOA 2021 Code": "LSOA code 2021"}, inplace=True)

print("Step 1 done")

# 2. Compute Population Metrics
pop_all["Total population"] = pop_all["Total"]

def sum_age_group(df, min_age, max_age):
    cols = []
    for age in range(min_age, max_age+1):
        col_f = f"F{age}"
        col_m = f"M{age}"
        if col_f in df.columns:
            cols.append(col_f)
        if col_m in df.columns:
            cols.append(col_m)
    if cols:
        return df[cols].sum(axis=1)
    else:
        return np.nan

# Calculate sums for desired age groups.
pop_all["Age <5"]    = sum_age_group(pop_all, 1, 4)
pop_all["Age 5-14"]  = sum_age_group(pop_all, 5, 14)
pop_all["Age 15-24"] = sum_age_group(pop_all, 15, 24)
pop_all["Age 25-64"] = sum_age_group(pop_all, 25, 64)
pop_all["Age 65+"]   = sum_age_group(pop_all, 65, 120)

# Total male population and percentage of males.
male_cols = [col for col in pop_all.columns if col.startswith("M")]
pop_all["Total_male"] = pop_all[male_cols].sum(axis=1)
pop_all["Percentage of males"] = (pop_all["Total_male"] / pop_all["Total population"]) * 100

print("Step 2 done")
print(pop_all.head())

# 3. Prepare June Population Data for Each Year
pop_june = pop_all.copy()
pop_june["Month"] = 6

pop_columns = [
    "LSOA code 2021", "Year", "Month", "Total population",
    "Age <5", "Age 5-14", "Age 15-24", "Age 25-64", "Age 65+", "Percentage of males"
]
pop_june = pop_june[pop_columns]

print("Step 3 done")

# 4. Merge the Population Data with the Baseline Dataset

baseline = pd.read_csv("C:/Users/20231441/OneDrive - TU Eindhoven/Documents/GitHub/MD-CBL-Group-12/data/Base/baseline_dataset.csv")
baseline["LSOA code 2021"] = baseline["LSOA code 2021"].astype(str)
pop_june["LSOA code 2021"] = pop_june["LSOA code 2021"].astype(str)
baseline = baseline.merge(pop_june, on=["LSOA code 2021", "Year", "Month"], how="left")

print("Step 4 done")

# 5. Interpolate and Extrapolate Population Data for Other Months

pop_vars = ["Total population", "Age <5", "Age 5-14", "Age 15-24", "Age 25-64", "Age 65+", "Percentage of males"]

# Create a continuous time variable (e.g., time = Year*12 + Month)
baseline["time"] = baseline["Year"] * 12 + baseline["Month"]

def interpolate_extrapolate(group):
    group = group.sort_values("time")
    for col in pop_vars:
        # Get available (non-missing) pairs
        mask = group[col].notna()
        if mask.sum() >= 2:
            x_known = group.loc[mask, "time"]
            y_known = group.loc[mask, col]
            # f will perform linear interpolation and extrapolation as needed.
            f = interp1d(x_known, y_known, kind="linear", fill_value="extrapolate", bounds_error=False)
            group[col] = f(group["time"])
    return group

baseline = baseline.groupby("LSOA code 2021").apply(interpolate_extrapolate)

print("Step 5 done")

# 6. For Age Groups: Compute Percentages of Total Population per LSOA per Month
for group in ["Age <5", "Age 5-14", "Age 15-24", "Age 25-64", "Age 65+"]:
    # Create a new column with percentage for this age group.
    pct_col = f"{group} (%)"
    baseline[pct_col] = (baseline[group] / baseline["Total population"]) * 100

# Normalize so the percentages sum to 100% (rounded to 2 decimals)
def normalize_age_pcts(row):
    cols = [f"Age <5 (%)", f"Age 5-14 (%)", f"Age 15-24 (%)", f"Age 25-64 (%)", f"Age 65+ (%)"]
    total = sum(row[col] for col in cols if pd.notnull(row[col]))
    if total > 0:
        for col in cols:
            row[col] = round(row[col] / total * 100, 2)
    return row

baseline = baseline.apply(normalize_age_pcts, axis=1)

print("Step 6 done")

# Save the Final Merged Baseline Population CSV #
baseline.to_csv("population.csv", index=False)
print(baseline.head())