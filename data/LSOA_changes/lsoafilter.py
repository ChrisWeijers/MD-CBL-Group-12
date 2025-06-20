import pandas as pd
from pathlib import Path

def main():
    data_dir = Path(__file__).resolve().parent.parent

    # 1. Load your full list of LSOAs
    all_lsoas = pd.read_csv(data_dir / "LSOA_changes/LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv")
    all_lsoas = all_lsoas.drop(columns=["LAD22NMW", "ObjectId"], errors="ignore")

    # 2. Filter to only Greater London area LSOAs
    london_lads = {
        "Camden", "Greenwich", "Hackney", "Hammersmith and Fulham", "Islington",
        "Kensington and Chelsea", "Lambeth", "Lewisham", "Southwark", "Tower Hamlets",
        "Wandsworth", "Westminster", "Barking and Dagenham", "Barnet", "Bexley",
        "Brent", "Bromley", "Croydon", "Ealing", "Enfield", "Haringey", "Harrow",
        "Havering", "Hillingdon", "Hounslow", "Kingston upon Thames", "Merton",
        "Newham", "Redbridge", "Richmond upon Thames", "Sutton", "Waltham Forest",
    }

    london_lsoas = all_lsoas[all_lsoas["LAD22NM"].isin(london_lads)].copy()

    print(f"Found {len(london_lsoas)} LSOAs in Greater London")

    # 3. Load the 2021 LSOA to ward data
    lsoa_to_ward = pd.read_csv(data_dir / "LSOA_changes/LSOA_(2021)_to_Electoral_Ward_(2024)_to_LAD_(2024)_Best_Fit_Lookup_in_EW.csv")
    lsoa_to_ward = lsoa_to_ward[["LSOA21CD", "WD24CD", "WD24NM"]]
    lsoa_to_ward = lsoa_to_ward.drop_duplicates(subset=["LSOA21CD"])

    # 4. Merge the two DataFrames on LSOA code
    london_lsoas = london_lsoas.merge(lsoa_to_ward, on="LSOA21CD", how="left")

    # 5. Save the resulting DataFrame to CSV
    london_lsoas.to_csv(data_dir / "LSOA_changes/london_lsoa11_lsoa21_lad22_ward24.csv", index=False)

    unique_count = london_lsoas["LSOA21CD"].nunique()
    print(f"Number of unique LSOA21CD entries: {unique_count}")

    print(london_lsoas.head())