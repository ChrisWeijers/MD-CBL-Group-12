import pandas as pd
import dash
from dash import dcc, html
import plotly.graph_objs as go

# Define the data
data = {
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
index = [
    "During the week",
    "At the weekend",
    "Morning/Afternoon",
    "Morning",
    "Afternoon",
    "Morning/afternoon (unsure which)",
    "Evening/Night",
    "Evening",
    "Early evening",
    "Late evening",
    "Night",
    "Evening/Night (unsure which)",
]

# Create initial DataFrame and transpose
df = pd.DataFrame(data, index=index)
df_T = df.T

# Compute average row
df_T.loc["Apr 2013 to Mar 2024"] = df_T.mean()
row = df_T.loc["Apr 2013 to Mar 2024"]

# Extract the reference values
total_week = row["During the week"]
total_weekend = row["At the weekend"]

# Define time labels
time_labels = [
    "Morning/Afternoon",
    "Morning",
    "Afternoon",
    "Morning/afternoon (unsure which)",
    "Evening/Night",
    "Evening",
    "Early evening",
    "Late evening",
    "Night",
    "Evening/Night (unsure which)",
]

clustered_data = []

# Create a separate bar for each label (grouped by time label)
for i, label in enumerate(time_labels):
    during_val = row[label] * total_week / 100
    weekend_val = row[label] * total_weekend / 100
    clustered_data.append(
        go.Bar(
            name=label,
            x=["During the week", "At the weekend"],
            y=[during_val, weekend_val],
            offsetgroup=str(i),  # Unique offset for clustering
        )
    )

# Dash app layout
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H2("Clustered Bar Chart: Time of Day Breakdown"),
    dcc.Graph(
        id='clustered-bar-chart',
        figure={
            'data': clustered_data,
            'layout': go.Layout(
                barmode='group',  # group bars side by side
                xaxis={'title': 'Day Type'},
                yaxis={'title': 'Percentage Breakdown'},
                legend={'x': 1, 'y': 1},
                margin={'b': 100}
            )
        }
    )
])

if __name__ == '__main__':
    app.run(debug=True)
