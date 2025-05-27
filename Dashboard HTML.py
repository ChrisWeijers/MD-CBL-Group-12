from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import geopandas as gpd
import streamlit as st
import pydeck as pdk
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import json

#Paths
geojson_path = r"C:\Users\20234783\Documents\GitHub\Data Challange 1 New\MD-CBL-Group-12\data\london_lsoa_combined.geojson" #geojson file location
data_path = r"C:\Users\20234783\Documents\GitHub\Data Challange 1 New\MD-CBL-Group-12\data\burglary_london.csv" #ofc later change to the path of the predicted stuff
#Load GeoJSON boundaries
gdf = gpd.read_file(geojson_path)
#Load burglary data
df = pd.read_csv(data_path, parse_dates=['Month'], low_memory=False)
#Count burglary incidents per LSOA code
burglary_counts = df.groupby('LSOA code').size().reset_index(name='Count')
merged = gdf.merge(burglary_counts, left_on="lsoa21cd", right_on="LSOA code", how="left")
merged['Count'] = merged['Count'].fillna(0)
merged = merged.to_crs(epsg=4326)
merged_json = json.loads(merged.to_json())

app = Dash(__name__)
app.layout = html.Div([
    html.H4('SOmething to show'),
    dcc.Graph(id="map"),
    dcc.Slider(0, 2000, 100,
               value=10,
               id='my-slider'
               ),
    html.Div(id='slider-output-container')
])

@app.callback(
    Output('slider-output-container', 'children'),
    Input('my-slider', 'value'))
def update_output(value):
    return 'You have selected "{}"'.format(value)



@app.callback(
    Output("map", "figure"),
    Input("map", "id"))
def display_choropleth(map):
    Output("graph", "figure"),
    Input("graph", "id"),
    Input('graph', 'clickData'))

def display_choropleth(graphid, graphclick):
    fig = px.choropleth(merged,
                        geojson=merged_json,
                        locations="lsoa21cd",
                        featureidkey='properties.lsoa21cd',
                        color="Count",
                        color_continuous_scale="Reds",
                        projection="mercator")
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    if graphclick is not None:
        print(f"loc: {graphclick['points'][0]['location']}\ncount: {graphclick['points'][0]['z']}")
    return fig

app.run(debug=True)
