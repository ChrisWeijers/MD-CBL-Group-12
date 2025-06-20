import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

def main():
    # Load the baseline dataset
    data_dir = Path(__file__).resolve().parent.parent
    baseline_file = data_dir / 'Base/baseline_dataset.csv'
    baseline = pd.read_csv(baseline_file)

    # Load, clean and map IMD data points
    # IMD data for 2008-06
    imd_2008 = pd.read_csv(data_dir / "IMD/imd2010adj_2015.csv", usecols=["LSOA code (2011)", "2010imd_rank"])
    imd_2008.rename(columns={
        "LSOA code (2011)": "LSOA code 2011",
        "2010imd_rank": "IMD Rank"
    }, inplace=True)

    imd_2008_mapped = pd.merge(imd_2008, baseline, on='LSOA code 2011', how='left')
    imd_2008_mapped["Year"] = 2008
    imd_2008_mapped["Month"] = 6

    # Split the data into two groups
    # Rows with indicator "U - Unchanged" or "S - Split"
    data_US = imd_2008_mapped[imd_2008_mapped["Change Indicator"].isin(["U", "S"])]
    data_US_reduced = data_US[["LSOA code 2021", "Year", "Month", "IMD Rank"]].copy()

    # Rows with indicator "M - Merged"
    data_M = imd_2008_mapped[imd_2008_mapped["Change Indicator"] == "M"]

    # Calculate the average qualifications for the LSOAs that were merged
    data_M_avg = (
        data_M.groupby(["LSOA code 2021", "Year", "Month"])["IMD Rank"]
              .mean()
              .reset_index()
    )
    data_M_avg.drop_duplicates(subset=["LSOA code 2021", "Year", "Month"], inplace=True)

    # Combine the two datasets back together
    combined_imd_2008 = pd.concat([data_US_reduced, data_M_avg], ignore_index=True)
    combined_imd_2008 = combined_imd_2008.drop_duplicates(subset=["LSOA code 2021", "Year", "Month"])

    # IMD data for 2012-06
    imd_2012 = pd.read_excel(data_dir / "IMD/File_1_ID_2015_Index_of_Multiple_Deprivation.xlsx", sheet_name="IMD 2015")
    imd_2012 = imd_2012[["LSOA code (2011)", "Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)"]]
    imd_2012.rename(columns={
        "LSOA code (2011)": "LSOA code 2011",
        "Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)": "IMD Rank"
    }, inplace=True)

    imd_2012_mapped = pd.merge(imd_2012, baseline, on='LSOA code 2011', how='left')
    imd_2012_mapped["Year"] = 2012
    imd_2012_mapped["Month"] = 6

    # Split the data into two groups
    # Rows with indicator "U - Unchanged" or "S - Split"
    data_US = imd_2012_mapped[imd_2012_mapped["Change Indicator"].isin(["U", "S"])]
    data_US_reduced = data_US[["LSOA code 2021", "Year", "Month", "IMD Rank"]].copy()

    # Rows with indicator "M - Merged"
    data_M = imd_2012_mapped[imd_2012_mapped["Change Indicator"] == "M"]

    # Calculate the average qualifications for the LSOAs that were merged
    data_M_avg = (
        data_M.groupby(["LSOA code 2021", "Year", "Month"])["IMD Rank"]
              .mean()
              .reset_index()
    )
    data_M_avg.drop_duplicates(subset=["LSOA code 2021", "Year", "Month"], inplace=True)

    # Combine the two datasets back together
    combined_imd_2012 = pd.concat([data_US_reduced, data_M_avg], ignore_index=True)
    combined_imd_2012 = combined_imd_2012.drop_duplicates(subset=["LSOA code 2021", "Year", "Month"])

    # IMD data for 2015-06
    imd_2015 = pd.read_excel(data_dir / "IMD/File_1_IMD2019_Index_of_Multiple_Deprivation.xlsx", sheet_name="IMD2019")
    imd_2015 = imd_2015[["LSOA code (2011)", "Index of Multiple Deprivation (IMD) Rank"]]
    imd_2015.rename(columns={
        "LSOA code (2011)": "LSOA code 2011",
        "Index of Multiple Deprivation (IMD) Rank": "IMD Rank"
    }, inplace=True)

    imd_2015_mapped = pd.merge(imd_2015, baseline, on='LSOA code 2011', how='left')
    imd_2015_mapped["Year"] = 2015
    imd_2015_mapped["Month"] = 6

    # Split the data into two groups
    # Rows with indicator "U - Unchanged" or "S - Split"
    data_US = imd_2015_mapped[imd_2015_mapped["Change Indicator"].isin(["U", "S"])]
    data_US_reduced = data_US[["LSOA code 2021", "Year", "Month", "IMD Rank"]].copy()

    # Rows with indicator "M - Merged"
    data_M = imd_2015_mapped[imd_2015_mapped["Change Indicator"] == "M"]

    # Calculate the average qualifications for the LSOAs that were merged
    data_M_avg = (
        data_M.groupby(["LSOA code 2021", "Year", "Month"])["IMD Rank"]
              .mean()
              .reset_index()
    )
    data_M_avg.drop_duplicates(subset=["LSOA code 2021", "Year", "Month"], inplace=True)

    # Combine the two datasets back together
    combined_imd_2015 = pd.concat([data_US_reduced, data_M_avg], ignore_index=True)
    combined_imd_2015 = combined_imd_2015.drop_duplicates(subset=["LSOA code 2021", "Year", "Month"])

    # Combine the three IMD data points into one DataFrame
    imd_data = pd.concat([combined_imd_2008, combined_imd_2012, combined_imd_2015], ignore_index=True)
    imd_data["LSOA code 2021"] = imd_data["LSOA code 2021"].astype(str)

    # Create an extended baseline that covers from 2008-06 to 2026-05
    lsoas = baseline["LSOA code 2021"].unique()
    extended_dates = pd.date_range(start="2008-06-01", end="2026-05-01", freq="MS")
    date_df = pd.DataFrame({
        "Year": extended_dates.year,
        "Month": extended_dates.month
    })
    lsoa_df = pd.DataFrame({"LSOA code 2021": lsoas})
    extended_baseline = lsoa_df.merge(date_df, how="cross")

    # Merge the extended baseline with the IMD data so that known IMD ranks are aligned to the correct month
    imd_extended = pd.merge(extended_baseline, imd_data, on=["LSOA code 2021", "Year", "Month"], how="left")

    # Estimate the IMD ranks for the missing months
    imd_extended["time"] = imd_extended["Year"] * 12 + imd_extended["Month"]
    imd_var = "IMD Rank"

    def interpolate_imd(group):
        group = group.sort_values("time")
        mask = group[imd_var].notna() & np.isfinite(group[imd_var])
        if mask.sum() >= 2:
            x_known = group.loc[mask, "time"]
            y_known = group.loc[mask, imd_var]
            f = interp1d(x_known, y_known, kind="linear", fill_value="extrapolate", bounds_error=False)
            new_vals = f(group["time"])
            new_vals = np.where(np.isfinite(new_vals), new_vals, y_known.iloc[-1])
            new_vals = np.clip(new_vals, 1, 32844)
            group[imd_var] = new_vals
        else:
            if mask.sum() > 0:
                group[imd_var] = np.clip(group.loc[mask, imd_var].iloc[0], 1, 32844)
        return np.round(group, 0)

    imd_interp = imd_extended.groupby("LSOA code 2021").apply(interpolate_imd)

    # Remove duplicate columns
    imd_interp = imd_interp.loc[:, ~imd_interp.columns.duplicated()]

    # Filter back to the baseline period and clean up the dataframe
    imd_final = imd_interp[imd_interp["Year"] >= 2011].copy()
    imd_final.drop(columns=['time'], inplace=True, errors='ignore')
    imd_final.reset_index(drop=True, inplace=True)

    # Save the finalized IMD data.
    imd_final.to_csv(data_dir / "IMD/imd_finalized.csv", index=False)
    print(imd_final.head())