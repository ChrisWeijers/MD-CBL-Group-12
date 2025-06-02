from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import geopandas as gpd
import json
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.enrich import DashProxy, Input, Output, html
from dash_extensions.javascript import arrow_function, assign


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

#Get_info feature for hovering
def get_info(feature=None):
    header = [html.H4("London Burglary Count")]
    if not feature:
        return header + [html.P("Hoover over a LSOA")]
    return header + [
        html.B(feature["properties"]["lsoa21nm"]),
        html.Br(),
        "{:.0f} Burglaries".format(feature["properties"]["Count"]),
    ]

#
classes = [0, 10, 20, 50, 100, 200, 350, 500]
colorscale = ["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026", "#800026"]
style = dict(weight=1, opacity=0.5, color="white", dashArray="1", fillOpacity=0.7)
# Create colorbar.
ctg = [
    "{}+".format(
        cls,
    )
    for i, cls in enumerate(classes[:-1])
] + ["{}+".format(classes[-1])]
colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale, width=300, height=30, position="bottomleft")
# Geojson rendering logic, must be JavaScript as it is executed in clientside.
style_handle = assign("""function(feature, context){
    const {classes, colorscale, style, colorProp} = context.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the color
    for (let i = 0; i < classes.length; ++i) {
        if (value > classes[i]) {
            style.fillColor = colorscale[i];  // set the fill color according to the class
        }
    }
    return style;
}""")
# Create geojson.
geojson = dl.GeoJSON(
    data=merged_json,  # url to geojson file
    style=style_handle,  # how to style each polygon
    zoomToBounds=True,  # when true, zooms to bounds when data changes (e.g. on load)
    zoomToBoundsOnClick=True,  # when true, zooms to bounds of feature (e.g. polygon) on click
    hoverStyle=arrow_function(dict(weight=5, color="black", dashArray="")),  # style applied on hover
    hideout=dict(colorscale=colorscale, classes=classes, style=style, colorProp="Count"),
    id="geojson",
)

# Create info control.
info = html.Div(
    children=get_info(),
    id="info",
    className="info",
    style={"position": "absolute", "top": "10px", "right": "10px", "zIndex": "1000"},
)

# Dash App
app = Dash(__name__)
app.layout = html.Div([
    html.H1('London Burglary Incidents Map'),
    dl.Map(
    children=[dl.TileLayer(), geojson, colorbar, info], style={"height": "60vh", "width": "60vw"}, center=[56, 10], zoom=6
    ),
    dcc.Graph(id="graph"),
    dcc.Slider(0, 2000, 100, value=10, id='my-slider'),
    html.Div(id='slider-output-container')
])

@app.callback(Output("info", "children"), Input("geojson", "hoverData"))
def info_hover(feature):
    return get_info(feature)

@app.callback(
    Output('slider-output-container', 'children'),
    Input('my-slider', 'value'))
def update_output(value):
    return f'You have selected "{value}"'

@app.callback(
    Output("graph", "figure"),
    Input("geojson", "clickData")
)
def update_line_chart(feature):
    if feature is not None:
        df_ = df[feature['properties']['lsoa21cd'] == df['LSOA code']]
        df_ = df_.groupby('Month').size().reset_index(name='Burglary Count')
        fig = px.line(df_, x=df_['Month'], y=df_['Burglary Count'], markers=True, labels={'x': "Months", 'y': "Burglaries"}, title='Monthly burglaries of selected LSOA')
        return fig
    else:
        df_ = df.groupby('Month').size().reset_index(name='Burglary Count')
        fig = px.line(df_, x=df_['Month'], y=df_['Burglary Count'], markers=True, labels={'x': "Months", 'y': "Burglaries"}, title='Monthly burglaries of London')
        return fig

if __name__ == "__main__":
    app.run(debug=True)
