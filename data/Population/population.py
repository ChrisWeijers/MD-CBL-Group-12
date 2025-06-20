import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

def main():
    data_dir = Path(__file__).resolve().parent.parent

    # 1. Load Population Data from Excel sheets
    pop_files = [
        data_dir / "Population/sapelsoasyoa20112014.xlsx",
        data_dir / "Population/sapelsoasyoa20152018.xlsx",
        data_dir / "Population/sapelsoasyoa20192022.xlsx"
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
    baseline_file = data_dir / 'Base/baseline_dataset.csv'
    baseline = pd.read_csv(baseline_file)
    baseline["LSOA code 2021"] = baseline["LSOA code 2021"].astype(str)
    pop_june["LSOA code 2021"] = pop_june["LSOA code 2021"].astype(str)
    population = baseline.merge(pop_june, on=["LSOA code 2021", "Year", "Month"], how="left")

    print("Step 4 done")

    # 5. Interpolate and Extrapolate Population Data for Other Months

    pop_vars = ["Total population", "Age <5", "Age 5-14", "Age 15-24", "Age 25-64", "Age 65+", "Percentage of males"]

    # Create a continuous time variable (e.g., time = Year*12 + Month)
    population["time"] = population["Year"] * 12 + population["Month"]

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

    population = population.groupby("LSOA code 2021").apply(interpolate_extrapolate)

    print("Step 5 done")

    # 6. For Age Groups: Compute Percentages of Total Population per LSOA per Month

    for group in ["Age <5", "Age 5-14", "Age 15-24", "Age 25-64", "Age 65+"]:
        # Create a new column with percentage for this age group.
        pct_col = f"{group} (%)"
        population[pct_col] = (population[group] / population["Total population"]) * 100

    # Function to round percentages while preserving the sum
    def round_preserve_sum(values, total=100.0, decimals=2):
        factor = 10 ** decimals
        scaled = values * factor
        floored = np.floor(scaled)
        remainder = scaled - floored
        deficit = int(round(total * factor - floored.sum()))
        # Distribute the deficit to the entries with the highest remainders
        indices = np.argsort(-remainder)
        floored[indices[:deficit]] += 1
        return floored / factor

    # Normalize so the percentages sum to 100% (rounded to 2 decimals)
    def normalize_age_pcts(row):
        cols = [f"Age <5 (%)", f"Age 5-14 (%)", f"Age 15-24 (%)",
                f"Age 25-64 (%)", f"Age 65+ (%)"]
        # Calculate raw percentages so that they sum to 100
        total = sum(row[col] for col in cols if pd.notnull(row[col]))
        if total > 0:
            # Raw normalized values (floating point)
            raw_vals = np.array([row[col] / total * 100 for col in cols])
            # Round while strictly preserving the sum of 100
            norm_vals = round_preserve_sum(raw_vals, total=100.0, decimals=2)
            for i, col in enumerate(cols):
                row[col] = norm_vals[i]
        return row

    population = population.apply(normalize_age_pcts, axis=1)

    print("Step 6 done")

    # 7. Finalize the Population Data

    # Clean up the dataframe
    columns_to_drop = [
        "LSOA code 2011", "LSOA name 2021", "Change Indicator",
        "Age <5", "Age 5-14", "Age 15-24", "Age 25-64", "Age 65+",
        "time"
    ]
    population.drop(columns=columns_to_drop, errors="ignore", inplace=True)
    population["Total population"] = population["Total population"].round(0)
    population["Percentage of males"] = population["Percentage of males"].round(2)

    # Rename "Percentage of males" to "Males (%)"
    population.rename(columns={"Percentage of males": "Males (%)"}, inplace=True)

    print("Step 7 done")

    # Save the finalized population data to a CSV file
    population.to_csv(data_dir / "Population/population_finalized.csv", index=False)
    print(population.head())