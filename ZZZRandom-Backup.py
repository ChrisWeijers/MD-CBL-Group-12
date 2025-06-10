import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.enrich import DashProxy, Input, Output, html
from dash_extensions.javascript import arrow_function, assign
import geopandas as gpd
import json
import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px

# Load paths
geojson_path = r'C:\Users\20234783\Documents\GitHub\Data Challange 1 New\MD-CBL-Group-12\data\london_lsoa_combined.geojson' #geojson file location
# geojson_pathw = r"C:\Users\20234783\Documents\GitHub\Data Challange 1 New\MD-CBL-Group-12\data\assets\Wards_May_2024_Boundaries_UK_BSC_8498175397534686318.geojson" #geojson file location
data_path = 'data/lsoa_predictions.csv'
# dataw_path = 'data/ward_predictions.csv'
#Load GeoJSON boundaries
gdf = gpd.read_file(geojson_path)
# gdfw = gpd.read_file(geojson_pathw)

#Load burglary data
df = pd.read_csv(data_path, parse_dates=['Month'], low_memory=False)
# dfw = pd.read_csv(dataw_path, parse_dates=['Month'], low_memory=False)

#Count burglary incidents per LSOA code
burglary_counts = df.groupby('LSOA code 2021').sum(numeric_only=True).drop(columns=['Year']).reset_index()
merged = gdf.merge(burglary_counts, left_on="lsoa21cd", right_on="LSOA code 2021", how="left")
merged['Predicted burglary count'] = merged['Predicted burglary count'].fillna(0)
merged = merged.to_crs(epsg=4326)
merged_json = json.loads(merged.to_json())

#Count burglary incidents per ward code
# burglary_counts = dfw.groupby('Ward code 2024').sum(numeric_only=True).drop(columns=['Year']).reset_index()
# mergedw = gdfw.merge(burglary_counts, left_on="WD24CD", right_on="Ward code 2024", how="left")
# mergedw[('Predicted burglary count')] = mergedw['Predicted burglary count'].fillna(0)
# mergedw = mergedw.to_crs(epsg=4326)
# WARDSDF = json.loads(mergedw.to_json())
# WARDSDF['features'] = [ward for ward in WARDSDF['features'] if not ward['properties']['Ward code 2024'] == None]
# wards = [ward['properties']['Ward code 2024'] for ward in WARDSDF['features']]

# Use the formula with 3 hours
df["Predicted hours per month"] = df["Predicted burglary count"] * 3 + 10
df['Predicted hours per week'] = df['Predicted hours per month'] / 4

def get_info(feature=None):
    header = [html.H4("London Burglary Count")]
    if not feature:
        return header + [html.P("Hover over an area")]
    return header + [
        html.B(feature["properties"]["lsoa21cd"]),
        html.Br(),
        "{:.0f} Burglaries".format(feature["properties"]["Predicted burglary count"]),
    ]


classes = [0, 10, 20, 30, 40, 50, 60, 70]
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
    hideout=dict(colorscale=colorscale, classes=classes, style=style, colorProp="Predicted burglary count"),
    id="geojson",
)
# Create info control.
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

# LSOA and Ward info gets processed
def get_code_from_feature(feature):
    if "WD24CD" in feature["properties"]:
        return feature["properties"]["WD24CD"]
    elif "lsoa21cd" in feature["properties"]:
        return feature["properties"]["lsoa21cd"]
    return None


# Create app.
app = DashProxy(prevent_initial_callbacks=False)
app.layout = html.Div([
    html.H1("London Burglary App"),

    html.Div([
        dl.Map(
            children=[dl.TileLayer(), geojson, selected_geojson, colorbar, info],
            style={"height": "60vh", "width": "60vw"},
            center=[51, 0],
            zoom=6
        ),
        dcc.Graph(id="graph"),
        html.Button('Submit', id='submit-val', n_clicks=0),
        dcc.Slider(0, 100, 10, value=0, id='my-slider'),
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

]
    )


@app.callback(Output("info", "children"), Input("geojson", "hoverData"))
def info_hover(feature):
    return get_info(feature)

@app.callback(Output('slider-output-container', 'children'), Input('my-slider', 'value'))
def update_output(value):
    return f'You have selected "{value}"'

@app.callback(Output("graph", "figure"),
              Output('submit-val', 'n_clicks'),
              Input("geojson", "clickData"),
              Input('submit-val', 'n_clicks'))
def update_line_chart(feature, nclicks):
    if nclicks >= 0:
        print(nclicks)
        if feature is not None:
            lsoa_code = feature['properties'].get('lsoa21cd') or feature['properties'].get('WD24CD')
            df_ = df[df['LSOA code 2021'] == lsoa_code]
            df_ = df_.groupby('Month').size().reset_index(name='Burglary Count')
            fig = px.line(df_, x='Month', y='Burglary Count', markers=True, labels={'Month': "Months", 'Burglary Count': "Burglaries"}, title='Monthly burglaries of selected LSOA')
        else:
            df_ = df.groupby('Month').size().reset_index(name='Burglary Count')
            fig = px.line(df_, x='Month', y='Burglary Count', markers=True, labels={'Month': "Months", 'Burglary Count': "Burglaries"}, title='Monthly burglaries of London')
        return fig, 0



@app.callback(Output("Data-Table", "data"), Input("geojson", "clickData"))
def update_data_table(feature):
    if feature:
        lsoa_code = get_code_from_feature(feature)
        filtered_df = df[df["LSOA code 2021"] == lsoa_code]
        return filtered_df.head(10).to_dict("records")
    return df.head(10).to_dict("records")

@app.callback(Output("selected-geojson", "data"), Input("geojson", "clickData"))
def update_selected_lsoa(feature):
    if feature and "properties" in feature:
        selected_code = get_code_from_feature(feature)
        if selected_code:
            if selected_code in merged["lsoa21cd"].values:
                selected = merged[merged["lsoa21cd"] == selected_code]
            # elif selected_code in mergedw["WD24CD"].values:
            #     selected = mergedw[mergedw["WD24CD"] == selected_code]
        if not selected.empty:
            return {"type": "FeatureCollection", "features": json.loads(selected.to_json())["features"]}
    return {"type": "FeatureCollection", "features": []}

@app.callback(
    Output("geojson", "data"),
    Input("my-slider", "value")
)
def update_map_slider(threshold):
    filtered = merged[merged["Predicted burglary count"] >= threshold]
    return json.loads(filtered.to_json())

if __name__ == "__main__":
    app.run(debug=True)