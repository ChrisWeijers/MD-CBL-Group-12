import os
import dash
from dash import dcc, dash_table
from dash_extensions.enrich import DashProxy, html, Input, Output
from dash_extensions.javascript import arrow_function, assign
import dash_leaflet as dl
import dash_leaflet.express as dlx
import plotly.express as px
import pandas as pd
import geopandas as gpd
import json

# Load paths
geojson_path    = r"E:\Niya - Year 2\MD-CBL\Dashboard\data\london_lsoa_combined.geojson"
data_path       = r"E:\Niya - Year 2\MD-CBL\Dashboard\data\Burglary London (2010 - 2025).csv"
data_lsoa_path  = r"E:\Niya - Year 2\MD-CBL\Dashboard\data"

# Load and prepare data
gdf = gpd.read_file(geojson_path)
df  = pd.read_csv(data_path, parse_dates=['Month'], low_memory=False)

# Load predictions and prepare datetime
predictions_path = os.path.join(data_lsoa_path, "lsoa_predictions.csv")
pred_df = pd.read_csv(predictions_path)
pred_df['Month'] = pred_df['Month'].astype(int)
pred_df['MonthDate'] = pd.to_datetime(
    dict(year=pred_df['Year'], month=pred_df['Month'], day=1)
)

# Prepare data
burglary_counts = df.groupby('LSOA code').size().reset_index(name='Count')
merged = gdf.merge(burglary_counts, left_on="lsoa21cd", right_on="LSOA code", how="left")
merged['Count'] = merged['Count'].fillna(0)
merged = merged.to_crs(epsg=4326)
merged_json = json.loads(merged.to_json())

# Compute centroid and name mappings
lsoa_code_to_name = dict(zip(merged['lsoa21cd'], merged['lsoa21nm']))
# Centroids for zooming
lsoa_code_to_center = {
    row['lsoa21cd']: [row['geometry'].centroid.y, row['geometry'].centroid.x]
    for _, row in merged.iterrows()
}

# Hover info box
def get_info(feature=None):
    header = [html.H4("London Burglary Count")]
    if not feature:
        return header + [html.P("Hover over an LSOA")]
    props = feature.get("properties", {})
    return header + [
        html.B(props.get("lsoa21nm", "")),
        html.Br(),
        f"{props.get('Count', 0):.0f} Burglaries",
    ]

# Choropleth settings
classes    = [0, 10, 20, 50, 100, 200, 350, 500]
colorscale = ["#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026", "#800026"]
style      = dict(weight=1, opacity=0.5, color="white", dashArray="1", fillOpacity=0.7)
ctg        = [f"{c}+" for c in classes]
colorbar   = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale,
                                      width=300, height=30, position="bottomleft")

# Choropleth style handle
def_style = assign("""function(feature, context){
    const {classes, colorscale, style, colorProp} = context.hideout;
    const value = feature.properties[colorProp];
    for (let i = 0; i < classes.length; i++) {
        if (value > classes[i]) style.fillColor = colorscale[i];
    }
    return style;
}""")

# Create GeoJSON layers
geojson = dl.GeoJSON(
    data=merged_json,
    style=def_style,
    zoomToBounds=True,
    zoomToBoundsOnClick=True,
    hoverStyle=arrow_function(dict(weight=5, color="black", dashArray="")),
    hideout=dict(classes=classes, colorscale=colorscale,
                 style=style, colorProp="Count"),
    id="geojson"
)

# Hover info box
def create_info_div():
    return html.Div(
        children=get_info(),
        id="info",
        style={"position": "absolute", "top": "10px", "right": "10px", "zIndex": "1000"}
    )
info = create_info_div()

# Selected LSOA visual overlay
selected_geojson = dl.GeoJSON(
    data={"type": "FeatureCollection", "features": []},
    id="selected-geojson",
    style=dict(weight=3, color="blue", dashArray="5,5", fillOpacity=0.3)
)

# App layout
app = DashProxy(__name__)
app.layout = html.Div([
    html.H1("London Burglary Incidents Dashboard"),
    html.Div([
        dl.Map(
            children=[dl.TileLayer(), geojson, selected_geojson, colorbar, info],
            id="map",
            center=[51.5074, -0.1278],
            zoom=10,
            style={"height": "60vh", "width": "60vw"}
        ),
        html.Br(),
        html.Label("Select an LSOA:"),
        dcc.Dropdown(
            id='lsoa-dropdown',
            options=[{'label': 'All London', 'value': 'ALL'}] +
                    [{'label': lsoa_code_to_name[code], 'value': code} for code in lsoa_code_to_name],
            value='ALL'
        ),
        dcc.Graph(id="graph"),
        html.Br(),
        dcc.Slider(0, 2000, 100, value=10, id='my-slider'),
        html.Div(id='slider-output-container')
    ], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '65%'}),
    html.Div([
        html.H2("Burglary Table"),
        dash_table.DataTable(
            id='Data-Table',
            columns=[{"name": col, "id": col} for col in df.columns],
            page_size=10,
            sort_action='native',
            filter_action='native',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
        )
    ], style={'display': 'inline-block', 'verticalAlign': 'top',
               'width': '30%', 'marginLeft': '2%'})
])

# Callbacks
@app.callback(Output("info", "children"), Input("geojson", "hoverData"))
def info_hover(feature):
    return get_info(feature)

@app.callback(Output('slider-output-container', 'children'), Input('my-slider', 'value'))
def update_output(value):
    return f'You have selected "{value}"'

@app.callback(Output('graph', 'figure'), Input('lsoa-dropdown', 'value'))
def update_line_chart(selected_lsoa):
    # Generate full date range from predictions data
    all_months = pd.date_range(
        start=pred_df['MonthDate'].min(),
        end=pred_df['MonthDate'].max(),
        freq="MS"
    )

    if selected_lsoa == 'ALL':
        df_sel = pred_df.copy()
        title = f'Predicted Burglaries for All London\n({all_months.min().strftime("%b %Y")}–{all_months.max().strftime("%b %Y")})'
    else:
        df_sel = pred_df[pred_df['LSOA code 2021'] == selected_lsoa]
        title = (
            f'Predicted Burglaries – {lsoa_code_to_name[selected_lsoa]}\n'
            f'({all_months.min().strftime("%b %Y")}–{all_months.max().strftime("%b %Y")})'
        )

    # Summarize and align predictions across full timeline
    ts = (
        df_sel
        .groupby('MonthDate')['Predicted burglary count']
        .sum()
        .reindex(all_months, fill_value=0)
        .reset_index()
    )
    ts.columns = ['Month', 'Predicted Burglaries']

    # Generate the line chart
    fig = px.line(
        ts, x='Month', y='Predicted Burglaries', markers=True,
        labels={'Month': 'Month', 'Predicted Burglaries': 'Predicted Burglaries'},
        title=title,
    )

    # Format x-axis to show month and year ticks
    fig.update_xaxes(
        tickformat="%b %Y",
        tickangle=45,
        dtick="M1",
        title="Month"
    )
    fig.update_yaxes(title="Predicted Burglaries")
    fig.update_layout(margin=dict(l=40, r=40, t=80, b=40))

    return fig

@app.callback(Output('lsoa-dropdown', 'value'), Input('geojson', 'clickData'), prevent_initial_call=True)
def sync_dropdown(clickData):
    props = clickData.get('properties', {}) if clickData else {}
    return props.get('lsoa21cd', dash.no_update)

@app.callback(Output('map', 'center'), Input('lsoa-dropdown', 'value'), prevent_initial_call=True)
def update_center(code):
    return lsoa_code_to_center.get(code, [51.5074, -0.1278])

@app.callback(Output('map', 'zoom'), Input('lsoa-dropdown', 'value'), prevent_initial_call=True)
def update_zoom(_):
    return 14

@app.callback(Output('selected-geojson', 'data'), Input('lsoa-dropdown', 'value'), prevent_initial_call=True)
def highlight_lsoa(code):
    if code == 'ALL':
        return {"type": "FeatureCollection", "features": []}
    sel = merged[merged['lsoa21cd'] == code]
    feat = json.loads(sel.to_json())['features'][0]
    return {"type": "FeatureCollection", "features": [feat]}

@app.callback(Output('Data-Table', 'data'), Input('lsoa-dropdown', 'value'))
def update_table(code):
    if code == 'ALL':
        return df.head(10).to_dict('records')
    filtered = df[df['LSOA code'] == code]
    return filtered.head(10).to_dict('records')

if __name__ == "__main__":
    app.run(debug=True)
