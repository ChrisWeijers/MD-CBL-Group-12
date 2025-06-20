import optuna
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import numpy as np
import shap
from pathlib import Path
import seaborn as sns

print('Preparing data...')
# Load the baseline dataset
data_dir = Path(__file__).resolve().parent.parent / 'data/'
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
smoothed_burglaries = pd.read_csv(data_dir / 'Smoothed_burglaries/smoothed_burglaries_finalized.csv')
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

data = data.drop_duplicates(subset=['LSOA code 2021', 'Year', 'Month'], keep='first')

# Encode categorical variable for LSOA code while preserving the original
data['LSOA encoded'] = data['LSOA code 2021'].astype('category').cat.codes

# Remove rows with 'Year' more than 2025 and 'Month' more than 4
data = data[((data['Year'] < 2025) | ((data['Year'] == 2025) & (data['Month'] <= 4)))]

# Make sure the data is properly formatted
invalid_cols = ['LSOA code 2011', 'LSOA name 2021', 'Change Indicator']
data = data.drop(columns=invalid_cols, errors='ignore')

# Rename columns for XGBoost compatibility
data.columns = data.columns.astype(str).str.replace(r'<', 'less than', regex=True)
data.columns = data.columns.astype(str).str.replace(r'>', 'more than', regex=True)

print('Splitting train and test data...')
# Split the data into training (2010-2023) and testing (2024-2025) sets
train_data = data[data['Year'] <= 2023].copy()
test_data = data[data['Year'] >= 2024].copy()

# Prepare features (X) and target (y) for training and testing
X_train = train_data.drop(columns=['LSOA code 2021', 'Burglary count', 'Year', 'Month'])
y_train = train_data['Burglary count'].astype(int)
X_test = test_data.drop(columns=['LSOA code 2021', 'Burglary count', 'Year', 'Month'])
y_test = test_data['Burglary count'].astype(int)

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

    tscv = TimeSeriesSplit(n_splits=13)
    cv_scores = []

    for train_index, val_index in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

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
best_model.fit(X_train, y_train)

# Make predictions on the train set.
train_pred = best_model.predict(X_train)

# Evaluation Metrics.
train_mse = mean_squared_error(y_train, train_pred)
train_r2 = r2_score(y_train, train_pred)
print(f'Train MSE: {train_mse}')
print(f'Train R2: {train_r2}')
print('===============================================================')

# Make predictions on the test set.
y_pred = best_model.predict(X_test)

# Evaluation Metrics.
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Test MSE: {mse}')
print(f'Test R2: {r2}')
print('===============================================================')

print('Creating SHAP analysis...')
# SHAP Analysis
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

plt.rcParams.update({'font.size': 14})
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)

residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50, kde=True)
plt.axvline(0, color='r', linestyle='--')
plt.title('Residual Distribution')
plt.xlabel('Prediction Error (y_test - y_pred)')
plt.show()

# Add predictions to the test data and save to CSV
test_data['Predicted Incident Count'] = y_pred
test_data.rename(columns={'Predicted Incident Count': 'Predicted burglary count'}, inplace=True)
test_data.to_csv('lsoa_test_predictions.csv', index=False)

# Save the best model
# best_model.save_model('all_data_xgboost_model.json')
# print('Model saved.')

# # Load the mapping file and rename its columns
# mapping = pd.read_csv(data_dir / 'LSOA_changes/london_lsoa11_lsoa21_lad22_ward24.csv')
# mapping = mapping.rename(columns={
#     'LSOA21CD': 'LSOA code 2021',
#     'LSOA21NM': 'LSOA name 2021',
#     'WD24CD': 'Ward code 2024',
#     'WD24NM': 'Ward name 2024'
# })

# # Add LSOA name 2021 to the test data if not present
# lsoa_mapping = mapping[['LSOA code 2021', 'LSOA name 2021']]
# test_data = test_data.merge(lsoa_mapping, on='LSOA code 2021', how='left')

# # Save a CSV without attributes used to make the predictions
# selected_cols = test_data[['LSOA code 2021', 'LSOA name 2021', 'Year', 'Month', 'Predicted burglary count']]
# selected_cols.to_csv('lsoa_predictions.csv', index=False)

# # Select the ward information columns directly from mapping
# ward_mapping = mapping[['LSOA code 2021', 'Ward code 2024', 'Ward name 2024']]

# # Merge the selected predictions with the ward mapping to get ward information
# ward_preds = selected_cols.merge(ward_mapping, on='LSOA code 2021', how='left')

# # Aggregate the predicted burglary counts to wards by Year and Month
# ward_aggr = ward_preds.groupby(['Ward code 2024', 'Ward name 2024', 'Year', 'Month'], as_index=False)['Predicted burglary count'].sum()
# ward_aggr = ward_aggr[['Ward code 2024', 'Ward name 2024', 'Year', 'Month', 'Predicted burglary count']]
# ward_aggr.to_csv('ward_predictions.csv', index=False)
# print('Predictions saved.')