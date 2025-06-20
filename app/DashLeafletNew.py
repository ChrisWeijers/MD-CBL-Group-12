import webbrowser
from threading import Timer

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
import DateTime as dt
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

base_dir = Path(__file__).resolve().parent.parent

geojson_path = base_dir / 'app/london_lsoa_combined.geojson'
geojson_pathw = base_dir / "app/Wards_May_2024_Boundaries_UK_BSC_8498175397534686318.geojson"
data_path = base_dir / "app/lsoa_predictions.csv"
dataw_path = base_dir / "app/ward_predictions.csv"

#Load GeoJSON boundaries
gdf = gpd.read_file(geojson_path)
gdfw = gpd.read_file(geojson_pathw)

#2021 lsoa to 2024 ward
l_to_w = pd.read_csv(
    base_dir / 'data/LSOA_changes/LSOA_(2021)_to_Electoral_Ward_(2024)_to_LAD_(2024)_Best_Fit_Lookup_in_EW.csv')
l_to_w_dict = dict(zip(l_to_w['LSOA21CD'], l_to_w['WD24CD']))

#City of London LSOAs
CofL = ['City of London 001A', 'City of London 001B', 'City of London 001C', 'City of London 001D', 'City of London 001E',
        'City of London 001F', 'City of London 001G']

#Load burglary data
df = pd.read_csv(data_path, parse_dates=['Month'], low_memory=False)
df['Month'] = [int(month) for month in df['Month']]
df['MonthDate'] = pd.to_datetime(dict(year=df['Year'], month=df['Month'], day=1))
dfw = pd.read_csv(dataw_path, parse_dates=['Month'], low_memory=False)
dfw['Month'] = [int(month) for month in dfw['Month']]
dfw['MonthDate'] = pd.to_datetime(dict(year=dfw['Year'], month=dfw['Month'], day=1))

#min + max time
min_time = min([val for val in dfw['MonthDate']])
max_time = max([val for val in dfw['MonthDate']])

#Unique sorted months for slider
months_in_df = df['MonthDate'].sort_values().unique()
labels = {i: f"{calendar.month_abbr[m.month]} {m.year}" for i, m in enumerate(months_in_df)}


#Count burglary incidents per LSOA code
burglary_counts = df.groupby('LSOA code 2021').sum(numeric_only=True).drop(columns=['Year', 'Month']).reset_index()
merged = gdf.merge(burglary_counts, left_on="lsoa21cd", right_on="LSOA code 2021", how="left")
merged = merged.merge(l_to_w, left_on='lsoa21cd', right_on='LSOA21CD', how='left').drop(columns=['msoa21cd','msoa21nm','lad22cd','lad22nm','LSOA21NMW','WD24NMW','LAD24CD','LAD24NM','LAD24NMW'])
merged['Predicted burglary count'] = merged['Predicted burglary count'].fillna(0)
merged = merged.to_crs(epsg=4326)
merged_json = json.loads(merged.to_json())
merged_json['features'] = [lsoa for lsoa in merged_json['features'] if not lsoa['properties']['lsoa21nm'] in CofL]

#Count burglary incidents per ward code
burglary_counts = dfw.groupby('Ward code 2024').sum(numeric_only=True).drop(columns=['Year', 'Month']).reset_index()
mergedw = gdfw.merge(burglary_counts, left_on="WD24CD", right_on="Ward code 2024", how="left")
mergedw[('Predicted burglary count')] = mergedw['Predicted burglary count'].fillna(0)
mergedw = mergedw.to_crs(epsg=4326)
WARDSDF = json.loads(mergedw.to_json())
WARDSDF['features'] = [ward for ward in WARDSDF['features'] if not ward['properties']['Ward code 2024'] == None]

#list of all wards
wards = [ward['properties']['Ward code 2024'] for ward in WARDSDF['features']]

# Formula (make this proportional)
df['WD24CD'] = df['LSOA code 2021'].map(l_to_w_dict)
merged_monthly = merged.merge(df[['LSOA code 2021', 'MonthDate', 'Predicted burglary count', 'WD24CD']],
                              left_on=['lsoa21cd', 'WD24CD'],
                              right_on=['LSOA code 2021', 'WD24CD'],
                              how='left')
merged_monthly.rename(columns={'Predicted burglary count_y': 'Predicted burglary count (LSOA)'}, inplace=True)
merged_monthly['Predicted burglary count'] = merged_monthly['Predicted burglary count (LSOA)'].fillna(0)
merged_monthly['Predicted burglary count'] = merged_monthly['Predicted burglary count'].fillna(0)
ward_monthly_totals = merged_monthly.groupby(['WD24CD', 'MonthDate'])['Predicted burglary count'].sum().reset_index()
ward_monthly_totals = ward_monthly_totals.rename(columns={'Predicted burglary count': 'Ward total predicted burglary count'})
merged_monthly = merged_monthly.merge(ward_monthly_totals, on=['WD24CD', 'MonthDate'], how='left')
merged_monthly['Ward total predicted burglary count'].replace(0, np.nan, inplace=True)
merged_monthly['Burglary proportion'] = merged_monthly['Predicted burglary count'] / merged_monthly['Ward total predicted burglary count']
merged_monthly['Recommended policing hours'] = merged_monthly['Burglary proportion'] * 3200
merged_monthly['Recommended policing hours'] = merged_monthly['Recommended policing hours'].fillna(0)
df["Recommended policing hours per month"] = merged_monthly['Recommended policing hours']
df['Recommended policing hours per week'] = df['Recommended policing hours per month'] / 4.5
dfw["Recommended policing hours per month"] = 3200
dfw['Recommended policing hours per week'] = dfw['Recommended policing hours per month'] / 4.5

#code to name dict for dropdown
code_to_name = dict(zip(merged['lsoa21cd'], merged['lsoa21nm']))
ward_code_to_name = dict(zip(merged['WD24CD'], merged['WD24NM']))
code_to_name.update(ward_code_to_name)

# Define the data
data_grouped_stacked_chart = {
    "Apr 2013 to Mar 2014": [67, 33, 41, 8, 23, 10, 59, 30, 20, 10, 24, 6],
    "Apr 2014 to Mar 2015": [68, 32, 41, 10, 23, 8, 59, 27, 16, 12, 22, 10],
    "Apr 2015 to Mar 2016": [66, 34, 39, 8, 22, 9, 61, 28, 18, 9, 27, 6],
    "Apr 2016 to Mar 2017": [70, 30, 43, 10, 23, 10, 57, 26, 16, 9, 27, 5],
    "Apr 2017 to Mar 2018": [70, 30, 39, 10, 20, 9, 61, 30, 19, 11, 26, 5],
    "Apr 2018 to Mar 2019": [67, 33, 42, 10, 23, 9, 58, 30, 21, 9, 23, 5],
    "Apr 2019 to Mar 2020": [70, 30, 38, 8, 20, 9, 62, 32, 19, 13, 26, 3],
    "Apr 2022 to Mar 2023": [71, 29, 32, 7, 19, 6, 68, 32, 19, 14, 28, 7],
    "Apr 2023 to Mar 2024": [69, 31, 40, 6, 29, 4, 60, 26, 15, 10, 33, 1],
}

# Define the index (row labels)
index_data_grouped_stacked_chart = [
    "Mon-Fri",
    "Sat-Sun",
    "Morning/Afternoon 2",
    "Morning",
    "Afternoon",
    "Morning or afternoon",
    "Evening/Night 2",
    "Evening",
    "Early evening",
    "Late evening",
    "Night",
    "Evening or Night",
]

# Create initial DataFrame and transpose
df_grouped_stacked_chart = pd.DataFrame(data_grouped_stacked_chart, index=index_data_grouped_stacked_chart)
df_grouped_stacked_chart_T = df_grouped_stacked_chart.T

# Compute average row
df_grouped_stacked_chart_T.loc["Apr 2013 to Mar 2024"] = df_grouped_stacked_chart_T.mean()
row_df_grouped_stacked_chart_T= df_grouped_stacked_chart_T.loc["Apr 2013 to Mar 2024"]

# Extract the reference values
total_week_df_grouped_stacked_chart_T = row_df_grouped_stacked_chart_T["Mon-Fri"]
total_weekend_df_grouped_stacked_chart_T = row_df_grouped_stacked_chart_T["Sat-Sun"]

# Define time labels to visualize
time_labels_df_grouped_stacked_chart_T = [
    "Morning",
    "Afternoon",
    "Morning or afternoon",
    "Early evening",
    "Late evening",
    "Night",
    "Evening or Night",
]

# Build clustered bar chart data
clustered_data_df_grouped_stacked_chart_T = []

for i, label in enumerate(time_labels_df_grouped_stacked_chart_T):
    during_val_df_grouped_stacked_chart_T = (row_df_grouped_stacked_chart_T[label] * total_week_df_grouped_stacked_chart_T / 100) / 5  # average per weekday
    weekend_val_df_grouped_stacked_chart_T = (row_df_grouped_stacked_chart_T[label] * total_weekend_df_grouped_stacked_chart_T / 100) / 2  # average per weekend day

    if label == "Late evening":
        early_during_df_grouped_stacked_chart_T = (row_df_grouped_stacked_chart_T["Early evening"] * total_week_df_grouped_stacked_chart_T / 100) / 5
        early_weekend_df_grouped_stacked_chart_T = (row_df_grouped_stacked_chart_T["Early evening"] * total_weekend_df_grouped_stacked_chart_T / 100) / 2

        clustered_data_df_grouped_stacked_chart_T.append(
            go.Bar(
                name="Late evening",
                x=["Mon-Fri", "Sat-Sun"],
                y=[during_val_df_grouped_stacked_chart_T, weekend_val_df_grouped_stacked_chart_T],
                base=[early_during_df_grouped_stacked_chart_T, early_weekend_df_grouped_stacked_chart_T],
                offsetgroup="evening-stack",
                legendgroup="evening-stack"
            )
        )
    elif label == "Early evening":
        clustered_data_df_grouped_stacked_chart_T.append(
            go.Bar(
                name="Early evening",
                x=["Mon-Fri", "Sat-Sun"],
                y=[during_val_df_grouped_stacked_chart_T, weekend_val_df_grouped_stacked_chart_T],
                offsetgroup="evening-stack",
                legendgroup="evening-stack"
            )
        )
    else:
        clustered_data_df_grouped_stacked_chart_T.append(
            go.Bar(
                name=label,
                x=["Mon-Fri", "Sat-Sun"],
                y=[during_val_df_grouped_stacked_chart_T, weekend_val_df_grouped_stacked_chart_T],
                offsetgroup=str(i),
            )
        )

# Get info that shows at the top right of the map
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

    # Top section: map + controls on the left, graph on the right
    html.Div([
        # Left panel (Map + Slider + Dropdown)
        html.Div([
            dl.Map(
                id='map',
                children=[dl.TileLayer(), geojson, colorbar, info],
                style={"height": "60vh", "width": "40vw"},
                center=[51, 0],
                zoom=10
            ),
            html.Br(),
            dcc.RangeSlider(
                min=0,
                max=len(months_in_df) - 1,
                step=1,
                value=[0, len(months_in_df) - 1],
                marks=labels,
                id='lsoa-date-range-slider'
            ),
            html.Br(),
            html.Label("Select an area:"),
            dcc.Dropdown(
                id='lsoa-dropdown',
                options=[{'label': 'All London', 'value': 'ALL'}] +
                        [{'label': code_to_name[code], 'value': code} for code in code_to_name],
                value='ALL'
            ),
            html.Div(id='Range shower'),
            html.Br(),
            html.Button('All LSOAs', id='lsoabutton', n_clicks=0),
            html.Button('All wards', id='wardbutton', n_clicks=0),
        ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

        # Right panel (Graph)
        html.Div([
            dcc.Graph(id="graph")
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
    ]),
    html.H2("Time of Day Breakdown (Average per Day 2013-2024)"),
    dcc.Graph(
        id='clustered-bar-chart',
        figure={
            'data': clustered_data_df_grouped_stacked_chart_T,
            'layout': go.Layout(
                barmode='group',
                xaxis={'title': 'Day'},
                yaxis={'title': 'Average Percentage per Day'},
                legend={'x': 1, 'y': 1},
                margin={'b': 100}
            )
        }
    ),
    # Bottom section: Data Table
    html.Div([
        html.H2("Burglary Table"),
        dash_table.DataTable(
            id='Data-Table',
            # columns=[{"name": col, "id": col} for col in df_dataframe.columns],
            columns=[],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
            sort_action='native',
            filter_action='native',
        )
    ], style={'padding': '20px'})
])


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
    Output("Data-Table", "data"),
    Output('lsoa-dropdown', 'value'),
    Output("Data-Table", "columns"),
    #    Output('selected-geojson', 'data'), #FOR SELECTION
    Input("geojson", "clickData"),
    Input("wardbutton", "n_clicks"),
    Input("lsoabutton", "n_clicks"),
    Input('lsoa-date-range-slider', 'value'),
    Input('lsoa-dropdown', 'value'),
    Input('geojson', 'data')
)
def update(feature, ward_clicks, lsoa_clicks, slider_range, dropdown, mapdata):

    #if the slider was used
    if slider_range:
        start_month = months_in_df[slider_range[0]]
        end_month = months_in_df[slider_range[1]]
        if start_month == min_time and end_month == max_time:
            filtered_df = df
            filtered_dfw = dfw
            filtered_json = merged_json
            filtered_jsonw = WARDSDF
        else:
            filtered_df = df[(df['MonthDate'] >= start_month) & (df['MonthDate'] <= end_month)]
            filtered_dfw = dfw[(dfw['MonthDate'] >= start_month) & (dfw['MonthDate'] <= end_month)]

            burglary_counts = filtered_df.groupby('LSOA code 2021').sum(numeric_only=True).reset_index()
            merged = gdf.merge(burglary_counts, left_on="lsoa21cd", right_on="LSOA code 2021", how="left")
            merged = merged.merge(l_to_w, left_on='lsoa21cd', right_on='LSOA21CD', how='left').drop(
                columns=['msoa21cd', 'msoa21nm', 'lad22cd', 'lad22nm', 'LSOA21NMW', 'WD24NMW', 'LAD24CD', 'LAD24NM',
                         'LAD24NMW'])
            merged['Predicted burglary count'] = merged['Predicted burglary count'].fillna(0)
            merged = merged.to_crs(epsg=4326)
            filtered_json = json.loads(merged.to_json())
            filtered_json['features'] = [lsoa for lsoa in merged_json['features'] if
                                       not lsoa['properties']['lsoa21nm'] in CofL]

            burglary_counts = filtered_dfw.groupby('Ward code 2024').sum(numeric_only=True).reset_index()
            mergedw = gdfw.merge(burglary_counts, left_on="WD24CD", right_on="Ward code 2024", how="left")
            mergedw[('Predicted burglary count')] = mergedw['Predicted burglary count'].fillna(0)
            mergedw = mergedw.to_crs(epsg=4326)
            filtered_jsonw = json.loads(mergedw.to_json())
            filtered_jsonw['features'] = [ward for ward in filtered_jsonw['features'] if
                                   not ward['properties']['Ward code 2024'] == None]

    #if something was clicked
    if feature:

        #if an lsoa was clicked
        if 'lsoa21cd' in feature['properties'].keys():
            ward = feature['properties']['WD24CD']
            df_1ward = filtered_json.copy()
            df_1ward['features'] = [lsoa for lsoa in df_1ward['features'] if lsoa['properties']['WD24CD'] == ward]


            # Generate full date range from predictions data
            all_months = pd.date_range(
                start=filtered_df['MonthDate'].min(),
                end=filtered_df['MonthDate'].max(),
                freq="MS"
            )

            df_sel = filtered_df[filtered_df['LSOA code 2021'] == feature['properties']['lsoa21cd']]
            title = (
                f"Predicted Burglaries – {code_to_name[feature['properties']['lsoa21cd']]}\n"
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

            # #show selected area FOR SELECTION
            # selected_code = get_code_from_feature(feature)
            # if selected_code:
            #     selected = merged[merged["lsoa21cd"] == selected_code]
            #     return {"type": "FeatureCollection", "features": json.loads(selected.to_json())["features"]}
            #
            return fig, df_1ward, None, df_sel.to_dict('records'), None, [{"name": col, "id": col} for col in df_sel.columns if col not in ['Month', 'Year']] #, {"type": "FeatureCollection", "features": json.loads(selected.to_json())["features"]}

        #if a ward was clicked
        else:
            ward = feature['properties']['WD24CD']
            df_1ward = filtered_json.copy()
            df_1ward['features'] = [lsoa for lsoa in df_1ward['features'] if lsoa['properties']['WD24CD'] == ward]

            # Generate full date range from predictions data
            all_months = pd.date_range(
                start=filtered_dfw['MonthDate'].min(),
                end=filtered_dfw['MonthDate'].max(),
                freq="MS"
            )

            df_sel = filtered_dfw[filtered_dfw['Ward code 2024'] == ward]
            title = (
                f"Predicted Burglaries – {code_to_name[ward]}\n"
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
            # df_ward = filtered_dfw[filtered_dfw['Ward code 2024'] == ward]
            # df_month = df_ward.groupby('Month').size().reset_index(name='Burglary Count')
            # fig = px.line(df_month, x=df_month['Month'], y=df_month['Burglary Count'], markers=True,
            #                 labels={'x': "Months", 'y': "Burglaries"}, title=f'Monthly predicted burglaries of selected ward ({feature["properties"]["WD24NM"]})')
            return fig, df_1ward, None, df_sel.to_dict('records'), None, [{"name": col, "id": col} for col in df_sel.columns if col not in ['Month', 'Year']]

    #if the dropdown was used
    if dropdown:

        #if dropdown selected all
        if dropdown == 'ALL':
            # Generate full date range from predictions data
            all_months = pd.date_range(
                start=filtered_df['MonthDate'].min(),
                end=filtered_df['MonthDate'].max(),
                freq="MS"
            )

            df_sel = filtered_df.copy()
            title = f'Predicted Burglaries for All London\n({all_months.min().strftime("%b %Y")}–{all_months.max().strftime("%b %Y")})'

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

            return fig, filtered_json, None, df_sel.to_dict('records'), None, [{"name": col, "id": col} for col in df_sel.columns if col not in ['Month', 'Year']]

        # if dropdown selected ward
        elif dropdown in wards:
            # Generate full date range from predictions data
            all_months = pd.date_range(
                start=filtered_dfw['MonthDate'].min(),
                end=filtered_dfw['MonthDate'].max(),
                freq="MS"
            )


            df_sel = filtered_dfw[filtered_dfw['Ward code 2024'] == dropdown]
            title = (
                f'Predicted Burglaries – {code_to_name[dropdown]}\n'
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

            #get ward map
            df_1ward = filtered_json.copy()
            df_1ward['features'] = [lsoa for lsoa in df_1ward['features'] if lsoa['properties']['WD24CD'] == dropdown]

            return fig, df_1ward, None, df_sel.to_dict('records'), None, [{"name": col, "id": col} for col in df_sel.columns if col not in ['Month', 'Year']]

        #if dropdown selected LSOA
        else:
            # Generate full date range from predictions data
            all_months = pd.date_range(
                start=filtered_df['MonthDate'].min(),
                end=filtered_df['MonthDate'].max(),
                freq="MS"
            )

            df_sel = filtered_df[filtered_df['LSOA code 2021'] == dropdown]
            title = (
                f'Predicted Burglaries – {code_to_name[dropdown]}\n'
                f'({all_months.min().strftime("%b %Y")}–{all_months.max().strftime("%b %Y")})'
            )

            # Summarize and align predictions across full timeline
            ts = (
                df_sel
                .groupby('MonthDate')['Predicted burglary count']
                .sum()
                .reindex(all_months,fill_value=0)
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

            #get ward map
            df_1ward = filtered_json.copy()
            ward = l_to_w_dict[dropdown]
            df_1ward['features'] = [lsoa for lsoa in df_1ward['features'] if lsoa['properties']['WD24CD'] == ward]

            return fig, df_1ward, None, df_sel.to_dict('records'), None, [{"name": col, "id": col} for col in df_sel.columns if col not in ['Month', 'Year']]

    #if nothing selected

    # Generate full date range from predictions data
    all_months = pd.date_range(
        start=filtered_df['MonthDate'].min(),
        end=filtered_df['MonthDate'].max(),
        freq="MS"
    )

    df_sel = filtered_df.copy()
    title = f'Predicted Burglaries for All London\n({all_months.min().strftime("%b %Y")}–{all_months.max().strftime("%b %Y")})'

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

    #check ward and LSOA buttons
    ctx = dash.callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "wardbutton":
        return fig, filtered_jsonw, None, filtered_dfw.to_dict('records'), None, [{"name": col, "id": col} for col in filtered_dfw.columns if col not in ['Month', 'Year']]#, {"type": "FeatureCollection", "features": []} #FOR SELECTION
    elif button_id == "lsoabutton":
        return fig, filtered_json, None, filtered_df.to_dict('records'), None, [{"name": col, "id": col} for col in filtered_df.columns if col not in ['Month', 'Year']]#, {"type": "FeatureCollection", "features": []} #FOR SELECTION
    else:
        if 'lsoa21cd' in mapdata['features'][0]['properties'].keys():
            return fig, filtered_json, None, df_sel.to_dict('records'), None, [{"name": col, "id": col} for col in df_sel.columns if col not in ['Month', 'Year']]#, {"type": "FeatureCollection", "features": []} #FOR SELECTION
        else:
            return fig, filtered_jsonw, None, filtered_dfw.to_dict('records'), None, [{"name": col, "id": col} for col in filtered_dfw.columns if col not in ['Month', 'Year']]#, {"type": "FeatureCollection", "features": []} #FOR SELECTION


if __name__ == "__main__":
    # Open the browser after a short delay
    Timer(1, lambda: webbrowser.open(f"http://127.0.0.1:8050")).start()

    app.run()
