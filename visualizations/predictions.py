import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

base_dir = Path(__file__).resolve().parent.parent
crime_file = base_dir / 'data/Crimes/crimes_finalized.csv'
df_crime = pd.read_csv(crime_file, usecols=['LSOA code 2021', 'Year', 'Month', 'Burglary count'])
df_pred = pd.read_csv(base_dir / 'app/lsoa_predictions.csv')
df_test = pd.read_csv(base_dir / 'XGBoost_Models/lsoa_test_predictions.csv', usecols=['LSOA code 2021', 'Year', 'Month', 'Predicted burglary count'])

def date_group(df, col='Predicted burglary count'):
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df['date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
    df = df.groupby('date')[[col]].sum()
    return df

df_crime = date_group(df_crime, 'Burglary count')
df_crime = df_crime[(df_crime.index >= '2011-01-01') & (df_crime.index <= '2025-04-01')]

df_pred = date_group(df_pred)
df_test = date_group(df_test)


fig = px.line(df_crime, x=df_crime.index, y='Burglary count',
              labels={'Burglary count': 'Burglary count', 'date': 'Date'})
fig.update_traces(name='Actual burglary count', showlegend=True)

# Add the prediction trace manually with specified color
fig.add_trace(go.Scatter(
    x=df_pred.index,
    y=df_pred['Predicted burglary count'],
    mode='lines',
    name='Predicted burglary count',
    line=dict(color='red')
))

fig.add_trace(go.Scatter(
    x=df_test.index,
    y=df_test['Predicted burglary count'],
    mode='lines',
    name='Test set predicted burglary count',
    line=dict(color='orange')
))

# Connect last actual to first predicted
fig.add_trace(go.Scatter(
    x=[df_crime.index[-1], df_pred.index[0]],
    y=[df_crime['Burglary count'].iloc[-1], df_pred['Predicted burglary count'].iloc[0]],
    mode='lines',
    name='Transition',
    line=dict(color='gray', dash='dot')  # Optional style
))

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.67
))

# Add shape regions
fig.add_vrect(
    x0="2011-01-01", x1="2024-01-01",
    fillcolor="blue", opacity=0.1,
    layer="below", line_width=0,
    name='Train set', showlegend=True
),


fig.add_vrect(
    x0="2024-01-01", x1="2025-04-01",
    fillcolor="orange", opacity=0.1,
    layer="below", line_width=0,
    name='Test set', showlegend=True
),

fig.add_vrect(
    x0="2025-04-01", x1="2026-05-01",
    fillcolor="red", opacity=0.1,
    layer="below", line_width=0,
    name='Predictions', showlegend=True
)

fig.show()


