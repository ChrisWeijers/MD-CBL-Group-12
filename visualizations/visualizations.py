import os
import ast
import pandas as pd
import matplotlib.pyplot as plt

#Config
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# File paths
BURGLARY_CSV = os.path.join(DATA_DIR, 'burglary_london.csv')
EXTRA_CSV    = os.path.join(DATA_DIR, 'extra_data.csv')

# Load burglary data
print(f"Loading burglary data from: {BURGLARY_CSV}")
df = pd.read_csv(BURGLARY_CSV, parse_dates=['Month'], low_memory=False)
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# 1. Time-series: Monthly burglary counts
monthly = df.groupby('Month').size().rename('Burglary Count')
fig, ax = plt.subplots(figsize=(12, 5))
monthly.plot(ax=ax)
ax.set_title('Monthly Burglary Counts in London (2010–2025)')
ax.set_xlabel('Month')
ax.set_ylabel('Number of Burglaries')
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, 'monthly_burglary_counts.png'))
plt.close(fig)
print("Saved plot: monthly_burglary_counts.png")

# 2. Bar chart: Top 10 LSOAs by total burglaries
lsoa_counts = df.groupby('LSOA name').size().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 6))
lsoa_counts.plot(kind='bar', ax=ax)
ax.set_title('Top 10 LSOAs by Total Burglary Count')
ax.set_xlabel('LSOA Name')
ax.set_ylabel('Total Burglaries')
ax.tick_params(axis='x', rotation=45)
for label in ax.get_xticklabels():
    label.set_ha('right')
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, 'top10_lsoas_burglary.png'))
plt.close(fig)
print("Saved plot: top10_lsoas_burglary.png")

# 3. Boxplot: Seasonal distribution by month
df['Month_Num']  = df['Month'].dt.month
month_names      = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
df['Month_Name'] = df['Month_Num'].apply(lambda m: month_names[m-1])

# Build boxplot data
box_data = []
for m in range(1, 13):
    annual_counts = (
        df[df['Month_Num'] == m]
        .groupby(df['Month'].dt.year)['Month']
        .count()
        .values
    )
    box_data.append(annual_counts)

fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(box_data, labels=month_names, showfliers=False)
ax.set_title('Annual Burglary Count Distribution by Month (2010–2025)')
ax.set_xlabel('Month')
ax.set_ylabel('Total Burglaries per Year')
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, 'seasonal_boxplot.png'))
plt.close(fig)
print("Saved plot: seasonal_boxplot.png")

# Load burglary totals
df_b = pd.read_csv(os.path.join(DATA_DIR, 'burglary_london.csv'), parse_dates=['Month'])
df_counts = df_b.groupby('LSOA code').size().rename('burglary_count').reset_index()

# Load & parse workforce shares
df_e = pd.read_csv(os.path.join(DATA_DIR, 'extra_data.csv'))
df_e['hours_list'] = df_e['hours_worked_percentages'].apply(ast.literal_eval)

# Expand into four separate columns
categories = {
    'PT ≤15h':   0,
    'PT 16–30h': 1,
    'FT 31–48h': 2,
    'OT ≥49h':   3,
}

for label, idx in categories.items():
    df_e[label] = df_e['hours_list'].apply(lambda lst: lst[idx])

#  Assign each LSOA its “dominant” category
df_e['dominant_hours'] = df_e[list(categories.keys())].idxmax(axis=1)

# Merge and sum burglaries by dominant category
df_merged = pd.merge(df_counts, df_e[['LSOA code','dominant_hours']],
                     on='LSOA code', how='inner')
totals = df_merged.groupby('dominant_hours')['burglary_count'].sum().reindex(categories.keys())

# Plot
fig, ax = plt.subplots(figsize=(8,5))
totals.plot(kind='bar', ax=ax, color=['C0','C1','C2','C3'])
ax.set_title('Total Burglaries by Dominant Work-Hours Category')
ax.set_xlabel('Work-Hours Category (dominant in each LSOA)')
ax.set_ylabel('Total Burglary Count')

for i, v in enumerate(totals):
    ax.text(i, v + 50, f"{v:,}", ha='center')
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, 'bar_dominant_workhours_burglary.png'))
plt.close(fig)
print("Saved: bar_dominant_workhours_burglary.png")

#Percentage of full-time working people against total burglaries per LSOA

# Load burglary totals
df_b = pd.read_csv(os.path.join(DATA_DIR, 'burglary_london.csv'), parse_dates=['Month'])
df_b['year'] = df_b['Month'].dt.year

df_counts = df_b.groupby('LSOA code').size().rename('burglary_count').reset_index()

# Load & parse workforce shares
df_e = pd.read_csv(os.path.join(DATA_DIR, 'extra_data.csv'))
df_e['hours_list'] = df_e['hours_worked_percentages'].apply(ast.literal_eval)

df_e['fulltime_percentage'] = df_e['hours_list'].apply(lambda x:x[2]) + df_e['hours_list'].apply(lambda x:x[3])
df_e = df_e[['fulltime_percentage', 'LSOA code']]
df_e = df_e.groupby(['LSOA code']).mean()


df_merged = pd.merge(df_counts, df_e, on='LSOA code', how='inner')

ax = plt.scatter(df_merged['fulltime_percentage'], df_merged['burglary_count'])
plt.xlabel('Percentage of people working full-time')
plt.ylabel('Total of burglaries')
plt.title('Total Burglaries by Percentage of People Working Full-time per LSOA')
plt.show()
