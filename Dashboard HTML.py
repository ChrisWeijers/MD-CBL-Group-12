from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import geopandas as gpd
import json
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.enrich import DashProxy
from dash_extensions.javascript import arrow_function, assign

# Load paths
geojson_path = r"C:\Users\20234783\Documents\GitHub\Data Challange 1 New\MD-CBL-Group-12\data\london_lsoa_combined.geojson"
data_path = r"C:\Users\20234783\Documents\GitHub\Data Challange 1 New\MD-CBL-Group-12\data\burglary_london.csv"

# Load and prepare data
gdf = gpd.read_file(geojson_path)
df = pd.read_csv(data_path, parse_dates=['Month'], low_memory=False)

# Prepare data
burglary_counts = df.groupby('LSOA code').size().reset_index(name='Count')
merged = gdf.merge(burglary_counts, left_on="lsoa21cd", right_on="LSOA code", how="left")
merged['Count'] = merged['Count'].fillna(0)
merged = merged.to_crs(epsg=4326)
merged_json = json.loads(merged.to_json())

# Hover info box
def get_info(feature=None):
    header = [html.H4("London Burglary Count")]
    if not feature:
        return header + [html.P("Hover over an LSOA")]
    return header + [
        html.B(feature["properties"]["lsoa21nm"]),
        html.Br(),
        f'{feature["properties"]["Count"]:.0f} Burglaries',
    ]

# Choropleth settings
classes = [0, 10, 20, 50, 100, 200, 350, 500]
colorscale = ["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026", "#800026"]
style = dict(weight=1, opacity=0.5, color="white", dashArray="1", fillOpacity=0.7)
ctg = [f"{c}+" for c in classes[:-1]] + [f"{classes[-1]}+"]
colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale, width=300, height=30, position="bottomleft")

style_handle = assign("""function(feature, context){
    const {classes, colorscale, style, colorProp} = context.hideout;
    const value = feature.properties[colorProp];
    for (let i = 0; i < classes.length; ++i) {
        if (value > classes[i]) {
            style.fillColor = colorscale[i];
        }
    }
    return style;
}""")

geojson = dl.GeoJSON(
    data=merged_json,
    style=style_handle,
    zoomToBounds=True,
    zoomToBoundsOnClick=True,
    hoverStyle=arrow_function(dict(weight=5, color="black", dashArray="")),
    hideout=dict(colorscale=colorscale, classes=classes, style=style, colorProp="Count"),
    id="geojson",
)

# Hover info box
info = html.Div(
    children=get_info(),
    id="info",
    className="info",
    style={"position": "absolute", "top": "10px", "right": "10px", "zIndex": "1000"},
)

# Selected LSOA visual overlay
selected_geojson = dl.GeoJSON(
    data={"type": "FeatureCollection", "features": []},
    id="selected-geojson",
    style=dict(weight=3, color="blue", dashArray="5,5", fillOpacity=0.3),
)

# App layout
app = DashProxy(__name__)
app.layout = html.Div([
    html.H1('London Burglary Incidents Map'),

    html.Div([
        dl.Map(
            children=[dl.TileLayer(), geojson, colorbar, info, selected_geojson],
            style={"height": "60vh", "width": "60vw"},
            center=[51.5074, -0.1278],  # Centered on London
            zoom=10
        ),
        dcc.Graph(id="graph"),
        dcc.Slider(0, 2000, 100, value=10, id='my-slider'),
        html.Div(id='slider-output-container')
    ], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '65%'}),

    html.Div([
        html.H2("Burglary Table"),
        dash_table.DataTable(
            id='Data-Table',
            columns=[{"name": col, "id": col} for col in df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
            page_size=10,
            sort_action='native',
            filter_action='native',
        )
    ], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '30%', 'marginLeft': '2%'})
])

# Callbacks
@app.callback(Output("info", "children"), Input("geojson", "hoverData"))
def info_hover(feature):
    return get_info(feature)

@app.callback(Output('slider-output-container', 'children'), Input('my-slider', 'value'))
def update_output(value):
    return f'You have selected "{value}"'

@app.callback(Output("graph", "figure"), Input("geojson", "clickData"))
def update_line_chart(feature):
    if feature and "properties" in feature:
        lsoa_code = feature['properties']['lsoa21cd']
        if lsoa_code:
            df_ = df[df['LSOA code'] == lsoa_code]
            df_ = df_.groupby('Month').size().reset_index(name='Burglary Count')
            return px.line(df_, x='Month', y='Burglary Count', markers=True,
                           labels={'Month': "Month", 'Burglary Count': "Burglaries"},
                           title=f'Monthly Burglaries for {lsoa_code}')
    # Default: all of London
    df_ = df.groupby('Month').size().reset_index(name='Burglary Count')
    return px.line(df_, x='Month', y='Burglary Count', markers=True,
                   labels={'Month': "Month", 'Burglary Count': "Burglaries"},
                   title='Monthly Burglaries in London')

@app.callback(Output("Data-Table", "data"), Input("geojson", "clickData"))
def update_data_table(feature):
    if feature:
        lsoa_code = feature["properties"]["lsoa21cd"]
        filtered_df = df[df["LSOA code"] == lsoa_code]
        return filtered_df.head(10).to_dict("records")
    return df.head(10).to_dict("records")

@app.callback(Output("selected-geojson", "data"), Input("geojson", "clickData"))
def update_selected_lsoa(feature):
    if feature and "properties" in feature:
        selected_code = feature["properties"]["lsoa21cd"]
        selected = merged[merged["lsoa21cd"] == selected_code]
        if not selected.empty:
            return {"type": "FeatureCollection", "features": json.loads(selected.to_json())["features"]}
    return {"type": "FeatureCollection", "features": []}

if __name__ == "__main__":
    app.run(debug=True)
