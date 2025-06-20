import pandas as pd
import numpy as np
from pathlib import Path

def main():
    data_dir = Path(__file__).resolve().parent.parent

    # Load the hours worked data for 2011 and clean it up
    hw_2011 = pd.read_csv(data_dir / 'Hours_worked/hours_worked_2011.csv')
    hw_2011 = hw_2011.rename(columns={
        '2011 super output area - lower layer': 'LSOA name 2011',
        'mnemonic': 'LSOA code 2011',
        'Part-time: 15 hours or less worked': '<= 15h worked (%)',
        'Part-time: 16 to 30 hours worked': '16h - 30h worked (%)',
        'Full-time: 31 to 48 hours worked': '31h - 48h worked (%)',
        'Full-time: 49 or more hours worked': '>= 49h worked (%)'
    })

    # Load the baseline dataset for mapping LSOA codes from 2011 to 2021
    baseline_file = data_dir / 'Base/baseline_dataset.csv'
    mapping = pd.read_csv(baseline_file)

    # Merge the mapping onto the education 2011 data.
    hw_2011_mapped = pd.merge(hw_2011, mapping, on='LSOA code 2011', how='left')
    hw_2011_mapped["Year"] = 2011
    hw_2011_mapped["Month"] = 1  # Assuming January for the month

    # Define the list of hours worked columns to process
    qual_cols = [
        '<= 15h worked (%)',
        '16h - 30h worked (%)',
        '31h - 48h worked (%)',
        '>= 49h worked (%)'
    ]

    # Split the data into two groups
    # Rows with indicator "U - Unchanged" or "S - Split"
    data_US = hw_2011_mapped[hw_2011_mapped["Change Indicator"].isin(["U", "S"])]
    data_US_reduced = data_US[["LSOA code 2021", "Year", "Month"] + qual_cols].copy()

    # Rows with indicator "M - Merged"
    data_M = hw_2011_mapped[hw_2011_mapped["Change Indicator"] == "M"]

    # Calculate the average hours worked for the LSOAs that were merged
    data_M_avg = (
        data_M.groupby(["LSOA code 2021", "Year", "Month"])[qual_cols]
              .mean()
              .reset_index()
    )
    data_M_avg.drop_duplicates(subset=["LSOA code 2021", "Year", "Month"], inplace=True)

    # Combine the two datasets back together
    combined_quals = pd.concat([data_US_reduced, data_M_avg], ignore_index=True)
    combined_quals = combined_quals.drop_duplicates(subset=["LSOA code 2021", "Year", "Month"])

    # Clean up the baseline dataset
    baseline = mapping.drop(columns=['LSOA code 2011', 'LSOA name 2021', 'Change Indicator'], errors='ignore')

    # Join the combined hours worked data with the baseline and clean it up
    hw_2011_finalized = pd.merge(baseline, combined_quals, on=["LSOA code 2021", "Year", "Month"], how="left")
    hw_2011_finalized = hw_2011_finalized.drop(columns=['LSOA code 2011', 'LSOA name 2021', 'Change Indicator'], errors='ignore')

    # Load the 2021 hours worked data and clean it up
    hw_2021 = pd.read_csv(data_dir / 'Hours_worked/hours_worked_2021.csv')
    hw_2021 = hw_2021.rename(columns={
        '2021 super output area - lower layer': 'LSOA name 2021',
        'mnemonic': 'LSOA code 2021',
        'Part-time: 15 hours or less worked': '<= 15h worked (%)',
        'Part-time: 16 to 30 hours worked': '16h - 30h worked (%)',
        'Full-time: 31 to 48 hours worked': '31h - 48h worked (%)',
        'Full-time: 49 or more hours worked': '>= 49h worked (%)'
    })
    hw_2021 = hw_2021.drop(columns=['LSOA name 2021'], errors='ignore')
    hw_2021["Year"] = 2021
    hw_2021["Month"] = 1  # Assuming January for the month

    # Join the 2021 hours worked data with the finalized 2011 data
    hours_worked_2011 = hw_2011_finalized.set_index(["LSOA code 2021", "Year", "Month"])
    hours_worked_2021 = hw_2021.set_index(["LSOA code 2021", "Year", "Month"])
    hours_worked_2011.update(hours_worked_2021)
    hw_2011_2021_finalized = hours_worked_2011.reset_index()

    # Function to round values while preserving the total sum to 100
    def round_preserve_sum(values, total=100.0, decimals=2):
        factor = 10 ** decimals
        scaled = values * factor
        floored = np.floor(scaled)
        remainder = scaled - floored
        deficit = int(round(total * factor - floored.sum()))
        # Distribute the deficit to the entries with highest remainders
        indices = np.argsort(-remainder)
        floored[indices[:deficit]] += 1
        return floored / factor

    # Function to perform the linear estimation for each LSOA
    def linear_estimation(group):
        row_2011 = group[(group['Year'] == 2011) & (group['Month'] == 1)]
        row_2021 = group[(group['Year'] == 2021) & (group['Month'] == 1)]

        y0 = row_2011[qual_cols].iloc[0].values.astype(float)
        y1 = row_2021[qual_cols].iloc[0].values.astype(float)

        # Compute the monthly slope (difference divided by 120 months)
        slope = (y1 - y0) / 120.0

        # Compute the estimated qualification percentages for a given row
        def predict(row):
            # Calculate t (months from January 2011)
            t = (row['Year'] - 2011) * 12 + (row['Month'] - 1)
            pred = y0 + slope * t
            # Clip negative predicted values to 0
            pred = np.maximum(pred, 0)
            # Normalize so the positive values sum to 100
            if pred.sum() != 0:
                pred_norm = pred / pred.sum() * 100
            else:
                pred_norm = pred
            # Round while strictly preserving the total of 100 proportionally to the values of each qualification category
            pred_final = round_preserve_sum(pred_norm, total=100.0, decimals=2)
            return pd.Series(pred_final, index=qual_cols)

        # Apply the prediction function to each row of the group
        group[qual_cols] = group.apply(predict, axis=1)
        # Compute the total (for verification that it equals 100)
        # group['Total'] = group[qual_cols].sum(axis=1)
        return group

    # Apply the linear estimation function to each LSOA
    df_est = hw_2011_2021_finalized.groupby("LSOA code 2021", group_keys=False).apply(linear_estimation)

    # Save the final estimated dataset.
    df_est.to_csv(data_dir / "Hours_worked/hours_worked_finalized.csv", index=False)
    print(df_est.head())