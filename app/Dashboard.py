import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk
import numpy as np
from shapely.geometry import Polygon, MultiPolygon

#Paths
geojson_path = "data/london_lsoa_combined.geojson" #geojson file location
data_path = "C:\\Users\\yongj\\OneDrive\\Desktop\\4CBLW00-20_Group_12\\data\\burglary_london.csv" #ofc later change to the path of the predicted stuff

#Load GeoJSON boundaries
gdf = gpd.read_file(geojson_path)

#Load burglary data
df = pd.read_csv(data_path, parse_dates=['Month'], low_memory=False)

#Count burglary incidents per LSOA code
burglary_counts = df.groupby('LSOA code').size().reset_index(name='Count')

#Merge with GeoDataFrame
merged = gdf.merge(burglary_counts, left_on="lsoa21cd", right_on="LSOA code", how="left")
merged['Count'] = merged['Count'].fillna(0)
merged = merged.to_crs(epsg=4326)

#Function to extract coordinates for Polygon and MultiPolygon
def get_polygon_coords(geom):
    if isinstance(geom, Polygon):
        return [list(coord) for coord in np.array(geom.exterior.coords)]
    elif isinstance(geom, MultiPolygon):
        return [
            [list(coord) for coord in np.array(p.exterior.coords)]
            for p in geom.geoms
        ]
    else:
        return []

#Get polygon coordinates for pydeck
merged['coordinates'] = merged.geometry.apply(get_polygon_coords)

#Define a color scale functiioon map count to a color gradient (green to red)
def count_to_color(count, max_count):
    norm = count / max_count if max_count > 0 else 0
    red = int(255 * norm)
    green = int(255 * (1 - norm))
    blue = 0
    alpha = 160
    return [red, green, blue, alpha]

max_count = merged['Count'].max()
merged['fill_color'] = merged['Count'].apply(lambda x: count_to_color(x, max_count))

#Limit data
merged_limited = merged.head(1500)

st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(
        latitude=merged_limited.geometry.centroid.y.mean(),
        longitude=merged_limited.geometry.centroid.x.mean(),
        zoom=10
    ),
    layers=[
        pdk.Layer(
            "PolygonLayer",
            data=merged_limited,
            get_polygon='coordinates',
            get_fill_color='fill_color',
            get_line_color=[0, 0, 0],
            line_width_min_pixels=1,
            pickable=True
        )],
    tooltip={"text": "{lsoa21nm}\nBurglary Count: {Count}"}
))