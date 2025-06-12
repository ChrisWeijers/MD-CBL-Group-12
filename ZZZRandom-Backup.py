import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.enrich import DashProxy, Input, Output, html
from dash_extensions.javascript import arrow_function, assign
import geopandas as gpd
import json
import pandas as pd
from dash import dcc, html, dash_table
import plotly.express as px
import calendar

# Load paths
geojson_path = r'C:\Users\20234783\Documents\GitHub\Data Challange 1 New\MD-CBL-Group-12\data\london_lsoa_combined.geojson'
data_path = 'data/lsoa_predictions.csv'

# Load GeoJSON boundaries
gdf = gpd.read_file(geojson_path)

# Load burglary data
df = pd.read_csv(data_path, parse_dates=['Month'], low_memory=False)

# Create 'MonthDate' column from Year and Month columns
df['Month'] = df['Month'].astype(int)
df['MonthDate'] = pd.to_datetime(dict(year=df['Year'], month=df['Month'], day=1))

# Extract unique sorted months for slider
months_in_df = df['MonthDate'].sort_values().unique()
labels = {i: f"{calendar.month_abbr[m.month]} {m.year}" for i, m in enumerate(months_in_df)}

# Count burglary incidents per LSOA code
burglary_counts = df.groupby('LSOA code 2021').sum(numeric_only=True).drop(columns=['Year']).reset_index()
merged = gdf.merge(burglary_counts, left_on="lsoa21cd", right_on="LSOA code 2021", how="left")
merged['Predicted burglary count'] = merged['Predicted burglary count'].fillna(0)
merged = merged.to_crs(epsg=4326)
merged_json = json.loads(merged.to_json())

# Formula with 3 hours
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

ctg = ["{}+".format(cls) for cls in classes[:-1]] + ["{}+".format(classes[-1])]
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
    hideout=dict(colorscale=colorscale, classes=classes, style=style, colorProp="Predicted burglary count"),
    id="geojson",
)

info = html.Div(
    children=get_info(),
    id="info",
    className="info",
    style={"position": "absolute", "top": "10px", "right": "10px", "zIndex": "1000"},
)

selected_geojson = dl.GeoJSON(
    data={"type": "FeatureCollection", "features": []},
    id="selected-geojson",
    style=dict(weight=3, color="blue", dashArray="5,5", fillOpacity=0.3),
)

def get_code_from_feature(feature):
    if "WD24CD" in feature["properties"]:
        return feature["properties"]["WD24CD"]
    elif "lsoa21cd" in feature["properties"]:
        return feature["properties"]["lsoa21cd"]
    return None

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
        dcc.RangeSlider(
            min=0,
            max=len(months_in_df)-1,
            step=1,
            value=[0, len(months_in_df)-1],
            marks=labels,
            tooltip={"placement": "bottom", "always_visible": True},
            id='lsoa-date-range-slider'
        ),
        html.Div(id='something'),
        dcc.Graph(id="graph"),
        html.Button('Submit', id='submit-val', n_clicks=0)
    ], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '65%'}),

    html.Div([
        html.H2("Burglary Table"),
        dash_table.DataTable(
            id='Data-Table',
            columns=[{"name": col, "id": col} for col in df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
            sort_action='native',
            filter_action='native',
        )
    ], style={'display': 'inline-block', 'verticalAlign': 'top', 'width': '30%', 'marginLeft': '2%'}),

])


@app.callback(
    Output('something', 'children'),
    Input('lsoa-date-range-slider', 'value')
)
def update_output(value):
    start_label = labels[value[0]]
    end_label = labels[value[1]]
    return f"Selected range: {start_label} to {end_label}"

@app.callback(Output("info", "children"), Input("geojson", "hoverData"))
def info_hover(feature):
    return get_info(feature)

@app.callback(
    Output("graph", "figure"),
    Input("geojson", "clickData"),
    Input('lsoa-date-range-slider', 'value'),
)
def update_line_chart(feature, slider_range):
    start_month = months_in_df[slider_range[0]]
    end_month = months_in_df[slider_range[1]]

    filtered_df = df[(df['MonthDate'] >= start_month) & (df['MonthDate'] <= end_month)]

    if feature is not None:
        lsoa_code = feature['properties'].get('lsoa21cd') or feature['properties'].get('WD24CD')
        filtered_df = filtered_df[filtered_df['LSOA code 2021'] == lsoa_code]

    plot_df = filtered_df.groupby('MonthDate').size().reset_index(name='Burglary Count')

    fig = px.line(
        plot_df,
        x='MonthDate',
        y='Burglary Count',
        markers=True,
        labels={'MonthDate': 'Month', 'Burglary Count': 'Burglaries'},
        title='Monthly burglaries of selected LSOA' if feature else 'Monthly burglaries of London'
    )
    return fig

@app.callback(
    Output("Data-Table", "data"),
    Input("geojson", "clickData"),
    Input('lsoa-date-range-slider', 'value'),
)
def update_data_table(feature, slider_range):
    start_month = months_in_df[slider_range[0]]
    end_month = months_in_df[slider_range[1]]

    filtered_df = df[(df['MonthDate'] >= start_month) & (df['MonthDate'] <= end_month)]

    if feature:
        lsoa_code = get_code_from_feature(feature)
        filtered_df = filtered_df[filtered_df["LSOA code 2021"] == lsoa_code]

    return filtered_df.to_dict("records")

@app.callback(Output("selected-geojson", "data"), Input("geojson", "clickData"))
def update_selected_lsoa(feature):
    if feature and "properties" in feature:
        selected_code = get_code_from_feature(feature)
        if selected_code:
            if selected_code in merged["lsoa21cd"].values:
                selected = merged[merged["lsoa21cd"] == selected_code]
                if not selected.empty:
                    return {"type": "FeatureCollection", "features": json.loads(selected.to_json())["features"]}
    return {"type": "FeatureCollection", "features": []}

if __name__ == "__main__":
    app.run(debug=True)
