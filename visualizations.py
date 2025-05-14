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

# ----------------------------------------
# Load burglary data
# ----------------------------------------
print(f"Loading burglary data from: {BURGLARY_CSV}")
df = pd.read_csv(BURGLARY_CSV, parse_dates=['Month'], low_memory=False)
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ----------------------------------------
# 1. Time-series: Monthly burglary counts
# ----------------------------------------
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

# ----------------------------------------
# 2. Bar chart: Top 10 LSOAs by total burglaries
# ----------------------------------------
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

# ----------------------------------------
# 3. Boxplot: Seasonal distribution by month
# ----------------------------------------
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

# ----------------------------------------
# Load extra socio-economic data
# ----------------------------------------
print(f"Loading extra data from: {EXTRA_CSV}")
df_extra = pd.read_csv(EXTRA_CSV)
# Parse list strings
df_extra['hours_worked_list']  = df_extra['hours_worked_percentages'].apply(ast.literal_eval)
df_extra['qual_list']          = df_extra['qualification_percentages'].apply(ast.literal_eval)

# Extract metrics: full-time 31–48h (index 2), Level 4+ quals (index 5)
df_extra['pct_fulltime_31_48']  = df_extra['hours_worked_list'].apply(lambda x: x[2])
df_extra['pct_qual_level4plus'] = df_extra['qual_list'].apply(lambda x: x[5])
df_extra['pop_density']         = df_extra['pop_density']

# Aggregate burglary counts by LSOA code
df_counts = df.groupby('LSOA code').size().rename('burglary_count').reset_index()

# Merge datasets
df_merged = pd.merge(
    df_counts,
    df_extra[['LSOA code', 'pop_density', 'pct_fulltime_31_48', 'pct_qual_level4plus']],
    on='LSOA code',
    how='inner'
)

# ----------------------------------------
# 4. Correlation scatter: Pop density vs burglaries
# ----------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_merged['pop_density'], df_merged['burglary_count'], s=10, alpha=0.6)
ax.set_title('Burglary Count vs Population Density')
ax.set_xlabel('Population Density (people/km²)')
ax.set_ylabel('Total Burglaries')
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, 'scatter_pop_density_vs_burglary.png'))
plt.close(fig)
print("Saved plot: scatter_pop_density_vs_burglary.png")

# ----------------------------------------
# 5. Correlation scatter: % Level 4+ quals vs burglaries
# ----------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_merged['pct_qual_level4plus'], df_merged['burglary_count'], s=10, alpha=0.6)
ax.set_title('Burglary Count vs % with Level 4+ Qualifications')
ax.set_xlabel('Percent with Level 4+ Qualifications')
ax.set_ylabel('Total Burglaries')
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, 'scatter_qual_level4_vs_burglary.png'))
plt.close(fig)
print("Saved plot: scatter_qual_level4_vs_burglary.png")

# ----------------------------------------
# 6. Correlation scatter: % Full-time (31-48h) vs burglaries
# ----------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_merged['pct_fulltime_31_48'], df_merged['burglary_count'], s=10, alpha=0.6)
ax.set_title('Burglary Count vs % Full-time (31-48h) Workers')
ax.set_xlabel('Percent Working 31-48 Hours')
ax.set_ylabel('Total Burglaries')
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, 'scatter_fulltime_vs_burglary.png'))
plt.close(fig)
print("Saved plot: scatter_fulltime_vs_burglary.png")

print("All visualizations saved in:", PLOTS_DIR)
