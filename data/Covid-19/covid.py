import pandas as pd

# Load the baseline dataset
baseline = pd.read_csv("C:/Users/20231441/OneDrive - TU Eindhoven/Documents/GitHub/MD-CBL-Group-12/data/Base/baseline_dataset.csv")

# Create the dummy variable for Covid-19
baseline["Covid-19 (boolean)"] = (((baseline["Year"] > 2020) | ((baseline["Year"] == 2020) & (baseline["Month"] >= 3)))).astype(int)

# Clean up the data
baseline = baseline.drop(columns=['LSOA code 2011', 'LSOA name 2021', 'Change Indicator'], errors="ignore")

# Save the data
baseline.to_csv("covid-19_finalized.csv", index=False)
print(baseline.head())