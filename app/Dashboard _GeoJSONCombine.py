import geopandas as gpd
import pandas as pd
import os

#Path to the folder with all the individual borough shapefiles
folder_path = ""  #Use the path for the file you downloaded from London dstabase

#Get list of all .shp files
shapefiles = [f for f in os.listdir(folder_path) if f.endswith('.shp')]

#Read and store all GeoDataFrames
gdfs = []
for shp in shapefiles:
    full_path = os.path.join(folder_path, shp)
    gdf = gpd.read_file(full_path)
    gdfs.append(gdf)

#Combine all borough GeoDataFrames into one
gdf_all = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

#New output path sonset in the folder or repo
output_path =r"C:\Users\yongj\OneDrive\Desktop\4CBLW00-20_Group_12\data\london_lsoa_combined.geojson"

#Save as GeoJSON
gdf_all.to_file(output_path, driver="GeoJSON")

print(f"Merged GeoJSON saved to: {output_path}")