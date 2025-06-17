import optuna
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import numpy as np
import shap
from pathlib import Path
import glob
import geopandas as gpd
from tqdm import tqdm

# -----------------------
# Load and Merge Datasets
# -----------------------

print('Preparing data...')
# Load the baseline dataset
base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / 'data/'
model_dir = base_dir / 'XGBoost_Models/'
baseline_file = data_dir / 'Base/baseline_dataset.csv'
baseline = pd.read_csv(baseline_file)
baseline.drop(columns=['LSOA code 2011', 'LSOA name 2021', 'Change Indicator'], inplace=True, errors='ignore')
baseline = baseline.drop_duplicates(subset=['LSOA code 2021', 'Year', 'Month'], keep='first')

# Load the other datasets
burglary_lag = pd.read_csv(data_dir / 'Burglary_lag/burglary_lag_finalized.csv')
covid = pd.read_csv(data_dir / 'Covid_19/covid-19_finalized.csv')
crimes = pd.read_csv(data_dir / 'Crimes/crimes_finalized.csv')
daylight = pd.read_csv(data_dir / 'Daylight/daylight_finalized.csv')
education = pd.read_csv(data_dir / 'Education/education_finalized.csv')
holidays = pd.read_csv(data_dir / 'Holidays_and_celebrations/holidays_finalized.csv')
hours_worked = pd.read_csv(data_dir / 'Hours_worked/hours_worked_finalized.csv')
household_income = pd.read_csv(data_dir / 'Household_income/household_income_finalized.csv')
imd = pd.read_csv(data_dir / 'IMD/imd_finalized.csv')
landuse = pd.read_csv(data_dir / 'Land_use/landuse_finalized.csv')
population = pd.read_csv(data_dir / 'Population/population_finalized.csv')
population_density = pd.read_csv(data_dir / 'Population_density/population_density_finalized.csv')
precipitation = pd.read_csv(data_dir / 'Precipitation/precipitation_finalized.csv')
smoothed_burglaries = pd.read_csv(data_dir / 'Smoothed burglaries/smoothed_burglaries_finalized.csv')
standard_deviation = pd.read_csv(data_dir / 'Standard_deviation/rolling_std_finalized.csv')
time_encoding = pd.read_csv(data_dir / 'Time_encoding/time_encoding_finalized.csv')

# Join all datasets
data = baseline.merge(burglary_lag, on=['Year','Month','LSOA code 2021'], how='left')
data = data.merge(covid, on=['Year', 'Month', 'LSOA code 2021'], how='left')
data = data.merge(crimes, on=['Year', 'Month', 'LSOA code 2021'], how='left')
data = data.merge(daylight, on=['Year', 'Month', 'LSOA code 2021'], how='left')
data = data.merge(education, on=['Year', 'Month', 'LSOA code 2021'], how='left')
data = data.merge(holidays, on=['Year', 'Month', 'LSOA code 2021'], how='left')
data = data.merge(hours_worked, on=['Year', 'Month', 'LSOA code 2021'], how='left')
data = data.merge(household_income, on=['Year', 'Month', 'LSOA code 2021'], how='left')
data = data.merge(imd, on=['Year', 'Month', 'LSOA code 2021'], how='left')
data = data.merge(landuse, on=['Year', 'Month', 'LSOA code 2021'], how='left')
data = data.merge(population, on=['Year', 'Month', 'LSOA code 2021'], how='left')
data = data.merge(population_density, on=['Year', 'Month', 'LSOA code 2021'], how='left')
data = data.merge(precipitation, on=['Year', 'Month', 'LSOA code 2021'], how='left')
data = data.merge(smoothed_burglaries, on=['Year', 'Month', 'LSOA code 2021'], how='left')
data = data.merge(standard_deviation, on=['Year', 'Month', 'LSOA code 2021'], how='left')
data = data.merge(time_encoding, on=['Year', 'Month', 'LSOA code 2021'], how='left')

# -------------------------------
# Data Cleaning and Preprocessing
# -------------------------------

data = data.drop_duplicates(subset=['LSOA code 2021', 'Year', 'Month'], keep='first')

# Encode categorical variable for LSOA code while preserving the original
data['LSOA encoded'] = data['LSOA code 2021'].astype('category').cat.codes

# Make sure the data is properly formatted
invalid_cols = ['LSOA code 2011', 'LSOA name 2021', 'Change Indicator']
data = data.drop(columns=invalid_cols, errors='ignore')

# Rename columns for XGBoost compatibility
data.columns = data.columns.astype(str).str.replace(r'<', 'less than', regex=True)
data.columns = data.columns.astype(str).str.replace(r'>', 'more than', regex=True)

# ------------------------------------------
# Split Data into Training and Forecast Sets
# ------------------------------------------

print('Splitting train and forecast data...')

train_df = data[(data["Year"]  < 2025) | ((data["Year"] == 2025) & (data["Month"] <= 4))].copy()
forecast_df = data[((data["Year"] == 2025) & (data["Month"] >= 5)) | (data["Year"] > 2025)].copy()

X_full = train_df.drop(columns=["LSOA code 2021","Burglary count","Year","Month"])
y_full = train_df["Burglary count"].astype(int)

# ----------------------------
# Train and Finetune the Model 
# ----------------------------

# Define the objective function for Optuna using TimeSeriesSplit
def objective(trial):
    params = {
        'objective': trial.suggest_categorical('objective', ['reg:squarederror', 'count:poisson']),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0, 1.0),
    }

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for train_index, val_index in tscv.split(X_full):
        X_tr, X_val = X_full.iloc[train_index], X_full.iloc[val_index]
        y_tr, y_val = y_full.iloc[train_index], y_full.iloc[val_index]

        model = XGBRegressor(random_state=42, **params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        cv_scores.append(mean_squared_error(y_val, preds))

    return np.mean(cv_scores)

# Create and run the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print('Best hyperparameters:', study.best_params)
print('Best CV MSE:', study.best_value)
print('===============================================================')

# Train the final model using the best parameters from Optuna
best_model = XGBRegressor(random_state=42, **study.best_params)
best_model.fit(X_full, y_full)

# Save the best model
best_model.save_model(model_dir / 'all_data_xgboost_model.json')
print('Model saved.')

# -------------
# SHAP Analysis
# -------------

print('Creating SHAP analysis...')
bg = X_full.sample(n=min(200, len(X_full)), random_state=0)
expl = shap.TreeExplainer(best_model, data=bg)
sv = expl.shap_values(X_full)
shap.summary_plot(sv, X_full, plot_type="bar", show=False)
plt.xlabel("Mean absolute SHAP value")
plt.savefig(model_dir / "new_shap.png", dpi=300)
plt.close()

# --------------------------------------------------------------------------
# Add missing crime and precipitation data by calculating the 5-year average
# --------------------------------------------------------------------------

print("Calculating missing crime and precipitation data...")

# Define the crime type columns you want to process.
data_types = ["Anti-social behaviour count", "Bicycle theft count", "Criminal damage and arson count",
               "Drugs count", "Other theft count", "Other crime count",
               "Possession of weapons count", "Public order count", "Robbery count", "Shoplifting count",
               "Theft from the person count", "Vehicle crime count", "Violence and sexual offences count",
               "Total precipitation (mm)", "Number of days (more than 1mm of precipitation)"]

train_grouped = (
    train_df
    .groupby(['LSOA code 2021', 'Month', 'Year'])[data_types]
    .mean()
    .sort_index()
)

for idx, row in tqdm(forecast_df.iterrows(),
                     total=forecast_df.shape[0],
                     desc="Imputing 5yr avg"):
    lsoa  = row["LSOA code 2021"]
    mo    = row["Month"]
    yr    = row["Year"]
    lo    = yr - 5
    hi    = yr - 1

    # Try to slice out the historical sub-DataFrame for this (lsoa, mo)
    try:
        hist_slice = train_grouped.loc[(lsoa, mo)]
        # hist_slice is a small DataFrame indexed by Year, columns = data_types
    except KeyError:
        # no history at all for this lsoa/month combo
        forecast_df.loc[idx, data_types] = np.nan
        continue

    # Now for each crime, just take the years lo…hi and mean
    for crime in data_types:
        # slice by year index and compute mean
        vals = hist_slice[crime].loc[lo:hi]
        avg_val = np.nan if vals.empty else round(vals.mean(), 0)
        forecast_df.at[idx, crime] = avg_val

# ---------------------------------------------------
# Prepare everything for calculating missing features
# ---------------------------------------------------

print("Preparing data for interative predictions...")

# Prepare historical pivot for burglary lags, EMAs, MSAs and STDs
hist = train_df.pivot_table(
    index=pd.to_datetime(train_df[["Year","Month"]].assign(DAY=1)),
    columns="LSOA code 2021",
    values="Burglary count",
    aggfunc="sum",
    fill_value=0
)

idx = pd.date_range(hist.index.min(), "2025-04-01", freq="MS")
hist = hist.reindex(idx, fill_value=0)

# Function to compute all time features for a single future row
def add_time_features(df, hist_df):
    # single‐row df at position idx 0
    i = df.index[0]
    d = pd.Timestamp(year=df.at[i, "Year"], month=df.at[i, "Month"], day=1)
    lsoa = df.at[i, "LSOA code 2021"]
    s_full = hist_df[lsoa]
    # shift to exclude current
    s = s_full.loc[:d].shift(1).fillna(0)

    # Lags
    df.at[i, "Burglary count (-1 month)"] = s.iloc[-1]
    df.at[i, "Burglary count (-2 month)"] = s_full.loc[:d].shift(2).fillna(0).iloc[-1]
    df.at[i, "Burglary count (-3 month)"] = s_full.loc[:d].shift(3).fillna(0).iloc[-1]

    # EMAs
    df.at[i, "Burglary count (12 month EMA)"] = s.ewm(span=12, min_periods=1, adjust=False).mean().iloc[-1]
    df.at[i, "Burglary count (6 month EMA)"]  = s.ewm(span=6,  min_periods=1, adjust=False).mean().iloc[-1]
    df.at[i, "Burglary count (3 month EMA)"]  = s.ewm(span=3,  min_periods=1, adjust=False).mean().iloc[-1]

    # SMAs
    df.at[i, "Burglary count (12 month SMA)"] = s.rolling(window=12, min_periods=1).mean().iloc[-1]
    df.at[i, "Burglary count (6 month SMA)"]  = s.rolling(window=6,  min_periods=1).mean().iloc[-1]
    df.at[i, "Burglary count (3 month SMA)"]  = s.rolling(window=3,  min_periods=1).mean().iloc[-1]

    # Rolling stds
    df.at[i, "Standard deviation (3 months)"]  = s.rolling(window=3,  min_periods=1).std(ddof=0).iloc[-1]
    df.at[i, "Standard deviation (6 months)"]  = s.rolling(window=6,  min_periods=1).std(ddof=0).iloc[-1]
    df.at[i, "Standard deviation (12 months)"] = s.rolling(window=12, min_periods=1).std(ddof=0).iloc[-1]

# Function to compute the smoothed burlgaries for a given window
def get_smoothed_value(hist_df, dt, lsoa, window,code_to_index, neighbor_dict, index_to_code):
    # Check target LSOA exists
    if lsoa not in hist_df.columns:
        return 0.0

    base = hist_df[lsoa].loc[:dt].shift(1).fillna(0)
    expected = base.rolling(window=window, min_periods=1).sum().iloc[-1] / window

    # Fetch its neighbour indices
    idx = code_to_index.get(lsoa)
    if idx is None:
        return expected

    neigh_idxs = neighbor_dict.get(idx, [idx])
    neigh_codes = [index_to_code[n] for n in neigh_idxs]

    # Collect valid neighbour values only
    vals = []
    for code in neigh_codes:
        if code not in hist_df.columns:
            continue # skip missing neighbour
        s_nb = hist_df[code].loc[:dt].shift(1).fillna(0)
        v = s_nb.rolling(window=window, min_periods=1).sum().iloc[-1] / window
        vals.append(v)

    # If no valid neighbours just return the target’s own value
    return float(np.mean(vals)) if vals else float(expected)

# Set the path to your LSOA shapefiles folder (adjust if needed)
shp_folder = data_dir / "LSOA_boundaries" / "LB_shp"
shp_files = glob.glob(str(shp_folder / "*.shp"))
gdf_list = [gpd.read_file(f) for f in shp_files]
if not gdf_list:
    raise ValueError(f"No shapefiles found in {shp_folder}")
gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))

# Rename LSOA column to match your other data
gdf = gdf.rename(columns={"lsoa21cd": "LSOA code 2021"})

# Reproject to British National Grid if needed
if gdf.crs is None or gdf.crs.to_string() != "EPSG:27700":
    gdf = gdf.to_crs("EPSG:27700")

# Add a numeric index for each geometry and keep only relevant columns
gdf = gdf.reset_index().rename(columns={"index": "geom_index"})
gdf = gdf[["geom_index", "LSOA code 2021", "geometry"]]

# Build spatial index and neighbor dictionary
sindex = gdf.sindex
neighbor_dict = {}
for idx, geom in gdf.geometry.items():
    candidates = list(sindex.intersection(geom.bounds))
    neighbors = [cand for cand in candidates if cand != idx and geom.touches(gdf.geometry.iloc[cand])]
    # Always include the geometry itself
    neighbors.append(idx)
    neighbor_dict[idx] = neighbors

# Build lookup dictionaries: from LSOA code to geom_index and vice versa
code_to_index = dict(zip(gdf["LSOA code 2021"], gdf["geom_index"]))
index_to_code = dict(zip(gdf["geom_index"], gdf["LSOA code 2021"]))

# ----------------------------------
# Iteratively forecast future months
# ----------------------------------

results = []
hist_df = hist.copy()

# sort forecast rows in chronological order
print('Predicting burglary counts...')
forecast_df = forecast_df.sort_values(["Year", "Month"])

forecast_df = forecast_df.sort_values(["Year", "Month", "LSOA code 2021"])

for _, row in tqdm(forecast_df.iterrows(), total=forecast_df.shape[0], desc="Forecast rows"):
    single = row.to_frame().T.copy()
    add_time_features(single, hist_df)
    Xp = single.drop(columns=["LSOA code 2021","Burglary count","Year","Month"], errors="ignore")
    Xp = Xp.apply(pd.to_numeric, errors='coerce')
    yhat = best_model.predict(Xp)[0]
    # record the full row parameters along with the prediction
    row_dict = single.iloc[0].to_dict()
    row_dict["Predicted burglary count"] = yhat

    # inject prediction into history for next lags
    dt = pd.Timestamp(year=row["Year"], month=row["Month"], day=1)
    if dt not in hist_df.index:
        hist_df.loc[dt] = 0
        hist_df.sort_index(inplace=True)
    hist_df.at[dt, row["LSOA code 2021"]] = yhat

    # compute smoothed burglary features
    smoothed_3  = get_smoothed_value(hist_df, dt, row["LSOA code 2021"], 3, code_to_index, neighbor_dict, index_to_code)
    smoothed_6  = get_smoothed_value(hist_df, dt, row["LSOA code 2021"], 6, code_to_index, neighbor_dict, index_to_code)
    smoothed_12 = get_smoothed_value(hist_df, dt, row["LSOA code 2021"], 12, code_to_index, neighbor_dict, index_to_code)

    row_dict["Smoothed burglaries (3 months)"]  = smoothed_3
    row_dict["Smoothed burglaries (6 months)"]  = smoothed_6
    row_dict["Smoothed burglaries (12 months)"] = smoothed_12

    results.append(row_dict)

# --------------------
# Save the predictions
# --------------------

print('Saving predictions...')
forecast = pd.DataFrame(results)
forecast = forecast.sort_values(["LSOA code 2021", "Year", "Month"])
forecast.to_csv(model_dir / "/lsoa_all_data_predictions.csv", index=False)

# Load the mapping file and rename its columns
mapping = pd.read_csv(data_dir / 'LSOA_changes/london_lsoa11_lsoa21_lad22_ward24.csv')
mapping = mapping.rename(columns={
    'LSOA21CD': 'LSOA code 2021',
    'LSOA21NM': 'LSOA name 2021',
    'WD24CD': 'Ward code 2024',
    'WD24NM': 'Ward name 2024'
})

# Add LSOA name 2021 to the test data
lsoa_mapping = mapping[['LSOA code 2021', 'LSOA name 2021']]
forecast = forecast.merge(lsoa_mapping, on='LSOA code 2021', how='left')

# Save a CSV without attributes used to make the predictions
selected_cols = forecast[['LSOA code 2021', 'LSOA name 2021', 'Year', 'Month', 'Predicted burglary count']]
selected_cols.to_csv(base_dir / 'app/lsoa_predictions.csv', index=False)

# Select the ward information columns directly from mapping
ward_mapping = mapping[['LSOA code 2021', 'Ward code 2024', 'Ward name 2024']]

# Merge the selected predictions with the ward mapping to get ward information
ward_preds = selected_cols.merge(ward_mapping, on='LSOA code 2021', how='left')

# Aggregate the predicted burglary counts to wards by Year and Month
ward_aggr = ward_preds.groupby(['Ward code 2024', 'Ward name 2024', 'Year', 'Month'], as_index=False)['Predicted burglary count'].sum()
ward_aggr = ward_aggr[['Ward code 2024', 'Ward name 2024', 'Year', 'Month', 'Predicted burglary count']]
ward_aggr.to_csv(base_dir / 'app/ward_predictions.csv', index=False)
print('Predictions saved.')