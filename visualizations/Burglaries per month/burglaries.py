import os
import glob
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Load the crimes data
crimes = pd.read_csv('C:/Users/20231441/OneDrive - TU Eindhoven/Documents/GitHub/MD-CBL-Group-12/data/Crimes/crimes_finalized.csv')
crimes = crimes[['LSOA code 2021', 'Year', 'Month', 'Burglary count']]

# Ensure Year and Month are integers and build a datetime column
crimes['Year'] = crimes['Year'].astype(int)
crimes['Month'] = crimes['Month'].astype(int)
crimes['date'] = pd.to_datetime(crimes[['Year', 'Month']].assign(DAY=1))

# Filter to relevant date range (2011-01 to 2025-02)
crimes = crimes[(crimes['date'] >= '2011-01-01') & (crimes['date'] <= '2025-02-01')]

# Load and combine LSOA boundary shapefiles (one per LAD)
shp_folder = "C:/Users/20231441/OneDrive - TU Eindhoven/Documents/GitHub/MD-CBL-Group-12/data/LSOA_boundaries/LB_shp"
shp_files = glob.glob(os.path.join(shp_folder, "*.shp"))
gdf_list = [gpd.read_file(shp_file) for shp_file in shp_files]
gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
gdf = gdf.rename(columns={'lsoa21cd': 'LSOA code 2021'})

# Create a date range for all months from 2011-01 to 2025-02
all_months = pd.date_range(start='2011-01-01', end='2025-02-01', freq='MS')

# Create an output directory for saved plots
output_dir = "C:/Users/20231441/OneDrive - TU Eindhoven/Documents/GitHub/MD-CBL-Group-12/visualizations/Burglaries per month/burglary plots"
os.makedirs(output_dir, exist_ok=True)

# For each month, merge crimes with geo data and plot
for current_date in all_months:
    crimes_month = crimes[crimes['date'] == current_date]
    gdf_merged = gdf.merge(crimes_month[['LSOA code 2021', 'Burglary count']], 
                           on='LSOA code 2021', how='left')
    gdf_merged['Burglary count'] = gdf_merged['Burglary count'].fillna(0)
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_merged.plot(column='Burglary count', ax=ax, legend=True, cmap="OrRd",
                    missing_kwds={"color": "lightgrey", "edgecolor": "red",
                                  "hatch": "///", "label": "Missing values"})
    ax.set_title(f"Burglaries for {current_date.strftime('%Y-%m')}", fontsize=16)
    ax.set_axis_off()
    
    # Save the figure to file
    output_path = os.path.join(output_dir, f"burglaries_{current_date.strftime('%Y_%m')}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)