import pandas as pd

population = pd.read_csv("population.csv")

# Remove unwanted columns. Using errors="ignore" in case any column is missing.
columns_to_drop = [
    "LSOA code 2011", "Change Indicator",
    "Age <5", "Age 5-14", "Age 15-24", "Age 25-64", "Age 65+",
    "time"
]
population.drop(columns=columns_to_drop, errors="ignore", inplace=True)

# Round "Total population" to 0 decimals and convert it to integer.
population["Total population"] = population["Total population"].round(0)

# Round "Percentage of males" to 2 decimal places.
population["Percentage of males"] = population["Percentage of males"].round(2)

# Rename "Percentage of males" to "Males (%)".
population.rename(columns={"Percentage of males": "Males (%)"}, inplace=True)
population.to_csv("population_finalized.csv", index=False)

print(population.head())