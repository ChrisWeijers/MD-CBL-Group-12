import pandas as pd
from pathlib import Path

def main():
    data_dir = Path(__file__).resolve().parent.parent
    baseline_file = data_dir / 'Base/baseline_dataset.csv'

    # Load the baseline dataset
    baseline = pd.read_csv(baseline_file)

    # Create the dummy variable for Covid_19
    baseline["Covid_19 (boolean)"] = (
                (baseline["Year"] > 2020) | ((baseline["Year"] == 2020) & (baseline["Month"] >= 3))).astype(int)

    # Clean up the data
    baseline = baseline.drop(columns=['LSOA code 2011', 'LSOA name 2021', 'Change Indicator'], errors="ignore")

    # Save the data
    baseline.to_csv(data_dir / "Covid_19/covid-19_finalized.csv", index=False)
    print(baseline.head())