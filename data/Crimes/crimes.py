import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Set the root folder that contains the yyyy-mm folders for the crime CSV files
root_dir = r'Original Dataset'  # Update this to your crime CSV root folder

# Path to the folder containing the borough shapefiles (each borough file contains its LSOAs)
borough_folder = r'C:/Users/20231441/OneDrive - TU Eindhoven/Documents/GitHub/MD-CBL-Group-12/data/LSOA boundaries/LB_shp'

# Load all shapefiles in the borough folder and combine them
shapefile_pattern = os.path.join(borough_folder, "*.shp")
shp_files = glob.glob(shapefile_pattern)
gdf_list = [gpd.read_file(shp) for shp in shp_files]
boroughs_gdf = pd.concat(gdf_list, ignore_index=True)
boroughs_gdf = gpd.GeoDataFrame(boroughs_gdf, crs=gdf_list[0].crs)
boroughs_gdf = boroughs_gdf.to_crs("EPSG:4326")

all_dfs = []

# Walk through each subfolder in the root directory
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        # Delete the file if it does NOT end with "metropolitan-street.csv"
        if not file.endswith("metropolitan-street.csv"):
            os.remove(file_path)
            print("Deleted:", file_path)
        else:
            print("Processing:", file_path)
            df = pd.read_csv(file_path)
            
            # If the coordinate columns exist, drop rows with missing coordinates.
            if "Longitude" in df.columns and "Latitude" in df.columns:
                df = df.dropna(subset=["Longitude", "Latitude"])
            else:
                print("Skipping file (missing coordinate columns):", file_path)
                continue

            # Create a geometry column from (Longitude, Latitude) pairs
            geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            
            # Spatial join with the combined borough boundaries using the fields "LSOA21NM" and "LSOA21CD"
            gdf_joined = gpd.sjoin(gdf, boroughs_gdf[['lsoa21nm', 'lsoa21cd', 'geometry']],
                                   how="left", predicate='within')
            
            # Update the "LSOA name" and "LSOA code" columns based on the spatial join results
            df["LSOA name"] = gdf_joined["lsoa21nm"]
            df["LSOA code"] = gdf_joined["lsoa21cd"]
            
            # Save the updated DataFrame back to CSV (overwriting the original file)
            df.to_csv(file_path, index=False)
            print("Updated LSOA information for:", file_path)
            
            # Append the processed DataFrame to the list for final combining
            all_dfs.append(df)

# After processing all files, combine all the DataFrames into one big DataFrame and save to CSV.
crime = pd.concat(all_dfs, ignore_index=True)

# Filter the crime data
crime = crime[['Month', 'LSOA code', 'Crime type']]
crime = crime.rename(columns={"LSOA code": "LSOA code 2021"})
crime[['Year', 'Month']] = crime['Month'].str.split('-', expand=True)
crime['Year'] = crime['Year'].astype(int)
crime['Month'] = crime['Month'].astype(int)
crime = crime[crime['Year'] >= 2011]
crime = crime.sort_values(["LSOA code 2021", "Year", "Month", "Crime type"])

# Aggregate the crime data to get incident counts for crime types per LSOA per year per month
crime['Incident Count'] = 1
agg_crime = (crime.groupby(["Year", "Month", "LSOA code 2021", "Crime type"], as_index=False)["Incident Count"].sum())

# Pivot the data to make each crime type a column
pivoted_data = agg_crime.pivot_table(
    index=['Year', 'Month', 'LSOA code 2021'],
    columns='Crime type',
    values='Incident Count',
    fill_value=0
).reset_index()

# Rename crime columns for clarity
pivoted_data.columns.name = None
pivoted_data = pivoted_data.rename_axis(None, axis=1)
pivoted_data.rename(
    columns=lambda col: col + " count" if col not in ["Year", "Month", "LSOA code 2021"] else col,
    inplace=True
)

# Load the baseline dataset
baseline = pd.read_csv('C:/Users/20231441/OneDrive - TU Eindhoven/Documents/GitHub/MD-CBL-Group-12/data/Base/baseline_dataset.csv')
baseline = baseline[['LSOA code 2021', 'Year', 'Month']]
baseline = baseline.sort_values(["LSOA code 2021", "Year", "Month"])

# Append the new data to the base data
full_data = pd.merge(baseline, pivoted_data, on=['LSOA code 2021', 'Year','Month'], how='left')
full_data = full_data.drop(columns=['Incident Count', 'Crime type'], errors='ignore')

# Fill in NA values with 0 for months before 2025-02
baseline_cols = ['Year', 'Month', 'LSOA code 2021']
crime_cols = [col for col in full_data.columns if col not in baseline_cols]
mask = ((full_data['Year'] < 2025) | ((full_data['Year'] == 2025) & (full_data['Month'] <= 2)))
full_data.loc[mask, crime_cols] = full_data.loc[mask, crime_cols].fillna(0)

# Save the final dataset to a new CSV file
full_data.to_csv('crimes_finalized.csv', index=False)
print(full_data.head())