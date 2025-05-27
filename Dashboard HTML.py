from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import geopandas as gpd
import json

# Paths
geojson_path = r"C:\Users\20234783\Documents\GitHub\Data Challange 1 New\MD-CBL-Group-12\data\london_lsoa_combined.geojson"
data_path = r"C:\Users\20234783\Documents\GitHub\Data Challange 1 New\MD-CBL-Group-12\data\burglary_london.csv"

# Load GeoJSON boundaries
gdf = gpd.read_file(geojson_path)

# Load burglary data
df = pd.read_csv(data_path, parse_dates=['Month'], low_memory=False)

# Count burglary incidents per LSOA code
burglary_counts = df.groupby('LSOA code').size().reset_index(name='Count')
merged = gdf.merge(burglary_counts, left_on="lsoa21cd", right_on="LSOA code", how="left")
merged['Count'] = merged['Count'].fillna(0)
merged = merged.to_crs(epsg=4326)
merged_json = json.loads(merged.to_json())

# Dash App
app = Dash(__name__)

app.layout = html.Div([
    html.H4('London Burglary Incidents Map'),
    dcc.Graph(id="map"),
    dcc.Slider(0, 2000, 100, value=10, id='my-slider'),
    html.Div(id='slider-output-container')
])

@app.callback(
    Output('slider-output-container', 'children'),
    Input('my-slider', 'value'))
def update_output(value):
    return f'You have selected "{value}"'

@app.callback(
    Output("map", "figure"),
    Input("my-slider", "value"))  # Trigger map update when slider changes
def display_choropleth(slider_value):
    # You can use the slider_value to filter data if needed
    fig = px.choropleth(
        merged,
        geojson=merged_json,
        locations="lsoa21cd",
        featureidkey='properties.lsoa21cd',
        color="Count",
        color_continuous_scale="Reds",
        projection="mercator"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
