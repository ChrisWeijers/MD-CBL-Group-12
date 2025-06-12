import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.enrich import DashProxy, Input, Output, html
from dash_extensions.javascript import arrow_function, assign
import geopandas as gpd
import json
import pandas as pd
import dash
from dash import dcc, dash_table
import plotly.express as px
import calendar

geojson_path = r'C:\Users\20234783\Documents\GitHub\Data Challange 1 New\MD-CBL-Group-12\data\london_lsoa_combined.geojson' #geojson file location
geojson_pathw = r"C:\Users\20234783\Documents\GitHub\Data Challange 1 New\MD-CBL-Group-12\data\Wards_May_2024_Boundaries_UK_BSC_8498175397534686318.geojson" #copy the path to the wards geojson file
data_path = r'C:\Users\20234783\Documents\GitHub\Data Challange 1 New\MD-CBL-Group-12\data\lsoa_predictions.csv'
dataw_path = r'C:\Users\20234783\Documents\GitHub\Data Challange 1 New\MD-CBL-Group-12\data\ward_predictions.csv'

#Load GeoJSON boundaries
gdf = gpd.read_file(geojson_path)
gdfw = gpd.read_file(geojson_pathw)

#2021 lsoa to 2024 ward
l_to_w = pd.read_csv(r'C:\Users\20234783\Documents\GitHub\Data Challange 1 New\MD-CBL-Group-12\data\LSOA_(2021)_to_Electoral_Ward_(2024)_to_LAD_(2024)_Best_Fit_Lookup_in_EW.csv')

#Load burglary data
df = pd.read_csv(data_path, parse_dates=['Month'], low_memory=False)
df['Month'] = [int(month) for month in df['Month']]
df['MonthDate'] = pd.to_datetime(dict(year=df['Year'], month=df['Month'], day=1))
dfw = pd.read_csv(dataw_path, parse_dates=['Month'], low_memory=False)
dfw['Month'] = [int(month) for month in dfw['Month']]
dfw['MonthDate'] = pd.to_datetime(dict(year=dfw['Year'], month=dfw['Month'], day=1))

#Unique sorted months for slider
months_in_df = df['MonthDate'].sort_values().unique()
labels = {i: f"{calendar.month_abbr[m.month]} {m.year}" for i, m in enumerate(months_in_df)}


#Count burglary incidents per LSOA code
burglary_counts = df.groupby('LSOA code 2021').sum(numeric_only=True).drop(columns=['Year']).reset_index()
merged = gdf.merge(burglary_counts, left_on="lsoa21cd", right_on="LSOA code 2021", how="left")
merged = merged.merge(l_to_w, left_on='lsoa21cd', right_on='LSOA21CD', how='left').drop(columns=['msoa21cd','msoa21nm','lad22cd','lad22nm','LSOA21NMW','WD24NMW','LAD24CD','LAD24NM','LAD24NMW'])
merged['Predicted burglary count'] = merged['Predicted burglary count'].fillna(0)
merged = merged.to_crs(epsg=4326)
merged_json = json.loads(merged.to_json())

#Count burglary incidents per ward code
burglary_counts = dfw.groupby('Ward code 2024').sum(numeric_only=True).drop(columns=['Year']).reset_index()
mergedw = gdfw.merge(burglary_counts, left_on="WD24CD", right_on="Ward code 2024", how="left")
mergedw[('Predicted burglary count')] = mergedw['Predicted burglary count'].fillna(0)
mergedw = mergedw.to_crs(epsg=4326)
WARDSDF = json.loads(mergedw.to_json())
WARDSDF['features'] = [ward for ward in WARDSDF['features'] if not ward['properties']['Ward code 2024'] == None]

# Formula (make this proportional)
df["Recommended policing hours per month"] = df["Predicted burglary count"] * 3 + 10
df['Recommended policing hours per week'] = df['Recommended policing hours per month'] / 4.5
dfw["Recommended policing hours per month"] = dfw["Predicted burglary count"] * 3 + 10
dfw['Recommended policing hours per week'] = dfw['Recommended policing hours per month'] / 4.5

#TO DO inbouwen opties voor LSOA of ward
def get_info(feature=None):
    header = [html.H4("London Burglary Count")]
    if not feature:
        return header + [html.P("Hover over an area")]
    elif 'lsoa21cd' in feature['properties'].keys():
        return header + [
            html.B(feature["properties"]["lsoa21nm"]),
            html.Br(),
            "{:.0f} Burglaries".format(feature["properties"]["Predicted burglary count"]),
        ]
    else:
        return header + [
            html.B(feature["properties"]["WD24NM"]),
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

################################### Last looks
# selected_geojson = dl.GeoJSON(
#     data={"type": "FeatureCollection", "features": []},
#     id="selected-geojson",
#     style=dict(weight=3, color="blue", dashArray="5,5", fillOpacity=0.3),
# )


# Create app.
app = DashProxy(prevent_initial_callbacks=False)
app.layout = html.Div([
    html.H1("London Burglary App"),
    dl.Map(id='map',
        children=[dl.TileLayer(), geojson, colorbar, info],
        style={"height": "60vh", "width": "60vw"},
        center=[56, 10],
        zoom=6
    ),
    dcc.RangeSlider(
        min=0,
        max=len(months_in_df) - 1,
        step=1,
        value=[0, len(months_in_df) - 1],
        marks=labels,
        tooltip={"placement": "bottom", "always_visible": True},
        id='lsoa-date-range-slider'
    ),
    html.Div(id='Range shower'),
    dcc.Graph(id="graph"),
    html.Button('All LSOAs', id='lsoabutton', n_clicks=0),
    html.Button('All wards', id='wardbutton', n_clicks=0),
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
    ]
)

@app.callback(
    Output('Range shower', 'value'),
    Input('lsoa-date-range-slider', 'value')
)
def update_output(value):
    start_label = labels[value[0]]
    end_label = labels[value[1]]
    return f"Selected range: {start_label} to {end_label}"

@app.callback(
    Output("info", "children"),
    Input("geojson", "hoverData"))
def info_hover(feature):
    return get_info(feature)

#buttons
@app.callback(
    Output("graph", "figure"),
    Output("geojson", "data"),
    Output('geojson', 'clickData'),
    # Output("Data-Table", "data"),
    Input("geojson", "clickData"),
    Input("wardbutton", "n_clicks"),
    Input("lsoabutton", "n_clicks"),
    # Input('lsoa-date-range-slider', 'value'),
)
def update(feature, ward_clicks, lsoa_clicks, slider_range):

    # start_month = months_in_df[slider_range[0]]
    # end_month = months_in_df[slider_range[1]]

    # It needs to go in :) and be specific for lsoa and wards
    # filtered_df = df[(df['MonthDate'] >= start_month) & (df['MonthDate'] <= end_month)]

    # if feature:
    #     lsoa_code = get_code_from_feature(feature)
    #     filtered_df = filtered_df[filtered_df["LSOA code 2021"] == lsoa_code]
    #
    # return filtered_df.to_dict("records")

    #if something was clicked
    if feature:
        #if an lsoa was clicked
        if 'lsoa21cd' in feature['properties'].keys():
            ward = feature['properties']['WD24CD']
            df_1ward = merged_json.copy()
            df_1ward['features'] = [lsoa for lsoa in df_1ward['features'] if lsoa['properties']['WD24CD'] == ward]
            df_lsoa = df[df['LSOA code 2021'] == feature['properties']['lsoa21cd']]
            print(df_lsoa.to_string())
            df_month = df_lsoa.groupby('Month').size().reset_index(name='Burglary Count')
            fig = px.line(df_month, x=df_month['Month'], y=df_month['Burglary Count'], markers=True,
                            labels={'x': "Months", 'y': "Burglaries"}, title=f'Monthly predicted burglaries of selected LSOA ({feature["properties"]["lsoa21nm"]})')
            return fig, df_1ward, None
        #if a ward was clicked
        else:
            ward = feature['properties']['WD24CD']
            df_1ward = merged_json.copy()
            df_1ward['features'] = [lsoa for lsoa in df_1ward['features'] if lsoa['properties']['WD24CD'] == ward]
            df_ward = dfw[dfw['Ward code 2024'] == ward]
            print(df_ward.to_string())
            df_month = df_ward.groupby('Month').size().reset_index(name='Burglary Count')
            fig = px.line(df_month, x=df_month['Month'], y=df_month['Burglary Count'], markers=True,
                            labels={'x': "Months", 'y': "Burglaries"}, title=f'Monthly predicted burglaries of selected ward ({feature["properties"]["WD24NM"]})')
            return fig, df_1ward, None

    df_ = df.groupby('Month').size().reset_index(name='Burglary Count')
    fig = px.line(df_, x=df_['Month'], y=df_['Burglary Count'], markers=True, labels={'x': "Months", 'y': "Burglaries"}, title='Monthly predicted burglaries of London')

    #check ward and LSOA buttons
    ctx = dash.callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "wardbutton":
        return fig, WARDSDF, None
    elif button_id == "lsoabutton":
        return fig, merged_json, None
    else:
        raise dash.exceptions.PreventUpdate



if __name__ == "__main__":
    app.run(debug=True)