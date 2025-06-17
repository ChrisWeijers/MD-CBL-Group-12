import os
import glob
import geopandas as gpd
import pandas as pd
from pathlib import Path

# Path to the folder containing the borough shapefiles
data_dir = Path(__file__).resolve().parent.parent
borough_folder = data_dir / 'LSOA_boundaries/LB_shp'

# Load all shapefiles in the borough folder and combine them
shapefile_pattern = os.path.join(borough_folder, "*.shp")
shp_files = glob.glob(shapefile_pattern)
gdf_list = [gpd.read_file(shp) for shp in shp_files]
lsoas_gdf = pd.concat(gdf_list, ignore_index=True)
lsoas_gdf = gpd.GeoDataFrame(lsoas_gdf, crs=gdf_list[0].crs)
lsoas_gdf.rename(columns={"lsoa21cd": "LSOA code 2021"}, inplace=True)
lsoas_gdf = lsoas_gdf.to_crs("EPSG:27700")

# Path to the landuse polygons shapefile
landuse_shp = data_dir / "Land_use/greater-london-latest-free.shp/gis_osm_landuse_a_free_1.shp"
landuse_col = "fclass" 

# Path to the water polygons shapefile
water_shp = data_dir / "Land_use/greater-london-latest-free.shp/gis_osm_water_a_free_1.shp"

# Desired projected CRS
target_crs = "EPSG:27700"

# Load and reproject LSOA_boundaries
gdf_lsoa = lsoas_gdf[["LSOA code 2021", "geometry"]].copy()
if gdf_lsoa.crs.to_string() != target_crs:
    gdf_lsoa = gdf_lsoa.to_crs(target_crs)

# Load and process landuse shapefile
gdf_land = gpd.read_file(landuse_shp)
if gdf_land.crs.to_string() != target_crs:
    gdf_land = gdf_land.to_crs(target_crs)

# Fill any missing land‐use category as "unclassified"
gdf_land["category"] = gdf_land[landuse_col].fillna("unclassified")

# Load and process water shapefile
gdf_water = gpd.read_file(water_shp)
if gdf_water.crs.to_string() != target_crs:
    gdf_water = gdf_water.to_crs(target_crs)

# Collapse all water subcategories into a single category "water"
gdf_water["category"] = "water"

# Concatenate the landuse and water dataframes
gdf_land = gdf_land[["category", "geometry"]]
gdf_water = gdf_water[["category", "geometry"]]
gdf_use = pd.concat([gdf_land, gdf_water], ignore_index=True)
gdf_use = gpd.GeoDataFrame(gdf_use, crs=target_crs)

# Intersect LSOAs with combined landuse/water polygons
intersected = gpd.overlay(
    gdf_lsoa[["LSOA code 2021", "geometry"]],
    gdf_use[["category", "geometry"]],
    how="intersection"
)

intersected = intersected.rename(columns={"geometry": "geom_intersection"})
intersected = intersected.set_geometry("geom_intersection")
intersected["area_m2"] = intersected.geom_intersection.area

# Sum area by LSOA_code and category
agg = (
    intersected
    .groupby(["LSOA code 2021", "category"], as_index=False)["area_m2"]
    .sum()
    .rename(columns={"area_m2": "landuse_area_m2"})
)

# Compute each LSOA’s total area
gdf_lsoa["lsoa_area_m2"] = gdf_lsoa.geometry.area
lsoa_area = gdf_lsoa[["LSOA code 2021", "lsoa_area_m2"]]

# Merge total LSOA area into the aggregated table
agg = agg.merge(lsoa_area, on="LSOA code 2021", how="left")

# Compute percentage of each category within each LSOA
agg["pct_of_lsoa"] = (agg["landuse_area_m2"] / agg["lsoa_area_m2"]) * 100.0

# Sum all classified areas by LSOA
classified_sum = (
    agg
    .groupby("LSOA code 2021", as_index=False)["landuse_area_m2"]
    .sum()
    .rename(columns={"landuse_area_m2": "sum_classified_m2"})
)

# Merge back on total area
classified_sum = classified_sum.merge(lsoa_area, on="LSOA code 2021", how="left")

# Compute residual
classified_sum["residual_m2"] = (
    classified_sum["lsoa_area_m2"] - classified_sum["sum_classified_m2"]
)

# # Build a dataframe of these residuals per LSOA label as "unclassified"
# residual_df = classified_sum[["LSOA code 2021", "residual_m2", "lsoa_area_m2"]].copy()
# residual_df["pct_of_lsoa"] = (residual_df["residual_m2"] / residual_df["lsoa_area_m2"]) * 100.0
# residual_df = residual_df[residual_df["residual_m2"] > 0.0]
# residual_df = residual_df.rename(columns={"residual_m2": "landuse_area_m2"})
# residual_df["category"] = "unclassified"

# # Append these residual rows to the agg table
# agg = pd.concat(
#     [
#         agg,
#         residual_df[["LSOA code 2021", "category", "landuse_area_m2", "lsoa_area_m2", "pct_of_lsoa"]],
#     ],
#     ignore_index=True
# )

# Pivot to one row per LSOA with columns for each category
pivot = agg.pivot_table(
    index="LSOA code 2021",
    columns="category",
    values="pct_of_lsoa",
    fill_value=0.0
).reset_index()

pivot.columns.name = None

pivot = pivot.copy()
cat_cols = pivot.columns.drop("LSOA code 2021")

# Round to 2 decimals
pivot[cat_cols] = pivot[cat_cols].round(2)

# Combine categories as specified
# Sum meadow, heath and grass into "herbaceous"
pivot["herbaceous"] = pivot.get("meadow", 0) + pivot.get("grass", 0) + pivot.get("heath", 0)

# Sum farmland, farmyard, and orchard into "farm"
pivot["farm"] = pivot.get("farmland", 0) + pivot.get("farmyard", 0) + pivot.get("orchard", 0)

# Sum scrub and forest into "forest"
pivot["forest"] = pivot.get("scrub", 0) + pivot.get("forest", 0)

# Rename "allotments" to "garden"
if "allotments" in pivot.columns:
    pivot["garden"] = pivot["allotments"]

# Rename "nature_reserve" to "nature reserve"
if "nature_reserve" in pivot.columns:
    pivot["nature reserve"] = pivot["nature_reserve"]

# Rename "recreation_ground" to "recreation ground"
if "recreation_ground" in pivot.columns:
    pivot["recreation ground"] = pivot["recreation_ground"]

# Remove columns that have been merged and drop unnecessary columns
cols_to_drop = []
for col in ["meadow", "grass", "farmland", "farmyard", "orchard", "scrub", "allotments", "heath", "nature_reserve", "recreation_ground", "quarry"]:
    if col in pivot.columns:
        cols_to_drop.append(col)
pivot.drop(columns=cols_to_drop, inplace=True)

# Rename all columns except "LSOA code 2021"
new_columns = {}
for col in pivot.columns:
    if col != "LSOA code 2021":
        new_columns[col] = col.capitalize() + " area (%)"
pivot.rename(columns=new_columns, inplace=True)

# Reorder columns so that "LSOA code 2021" comes first
cols = pivot.columns.tolist()
cols.remove("LSOA code 2021")
pivot = pivot[["LSOA code 2021"] + cols]

# Load the baseline dataset
baseline_file = data_dir / 'Base/baseline_dataset.csv'
baseline = pd.read_csv(baseline_file, dtype={"LSOA code 2021": str})

# Merge the landuse data with the baseline dataset
merged = baseline.merge(pivot, on="LSOA code 2021", how="left")

# Save the merged result to CSV
merged.to_csv(data_dir / "Land_use/landuse_finalized.csv", index=False)
print(merged.head())