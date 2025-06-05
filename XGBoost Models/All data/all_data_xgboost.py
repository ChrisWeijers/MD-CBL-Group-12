import optuna
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import shap
from pathlib import Path

# Load the baseline dataset
data_dir = Path(__file__).resolve().parent.parent.parent / 'data/'
baseline_file = data_dir / 'Base/baseline_dataset.csv'
baseline = pd.read_csv(baseline_file)
baseline.drop(columns=['LSOA code 2011', 'LSOA name 2021', 'Change Indicator'], inplace=True, errors='ignore')
baseline = baseline.drop_duplicates(subset=['LSOA code 2021', 'Year', 'Month'], keep='first')

# Load the other datasets
burglary_lag = pd.read_csv(data_dir / 'Burglary lag/burglary_lag_finalized.csv')
covid = pd.read_csv(data_dir / 'Covid-19/covid-19_finalized.csv')
crimes = pd.read_csv(data_dir / 'Crimes/crimes_finalized.csv')
daylight = pd.read_csv(data_dir / 'Daylight/daylight_finalized.csv')
education = pd.read_csv(data_dir / 'Education/education_finalized.csv')
holidays = pd.read_csv(data_dir / 'Holidays and celebrations/holidays_finalized.csv')
hours_worked = pd.read_csv(data_dir / 'Hours worked/hours_worked_finalized.csv')
household_income = pd.read_csv(data_dir / 'Household income/household_income_finalized.csv')
imd = pd.read_csv(data_dir / 'IMD/imd_finalized.csv')
landuse = pd.read_csv(data_dir / 'Land use/landuse_finalized.csv')
population = pd.read_csv(data_dir / 'Population/population_finalized.csv')
population_density = pd.read_csv(data_dir / 'Population density/population_density_finalized.csv')
precipitation = pd.read_csv(data_dir / 'Precipitation/precipitation_finalized.csv')
smoothed_burglaries = pd.read_csv(data_dir / 'Smoothed burglaries/smoothed_burglaries_finalized.csv')

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

data = data.drop_duplicates(subset=['LSOA code 2021', 'Year', 'Month'], keep='first')

# Encode categorical variable for LSOA code while preserving the original
data['LSOA encoded'] = data['LSOA code 2021'].astype('category').cat.codes

# Remove rows with "Year" more than 2025 and "Month" more than 2
data = data[((data["Year"] < 2025) | ((data["Year"] == 2025) & (data["Month"] <= 2)))]

# Rename columns for XGBoost compatibility
data.columns = data.columns.astype(str).str.replace(r'<', 'less than', regex=True)
data.columns = data.columns.astype(str).str.replace(r'>', 'more than', regex=True)

# Idk which one but there is a dataset which isn't properly formatted
invalid_cols = ['LSOA code 2011', 'LSOA name 2021', 'Change Indicator']
data = data.drop(columns=invalid_cols, errors='ignore')
print(data.info())

# Split the data into training (2010-2023) and testing (2024-2025) sets
train_data = data[data['Year'] <= 2023].copy()
test_data = data[data['Year'] >= 2024].copy()

# Prepare features (X) and target (y) for training and testing
X_train = train_data.drop(columns=['LSOA code 2021', 'Burglary count'])
y_train = train_data['Burglary count']
X_test = test_data.drop(columns=['LSOA code 2021', 'Burglary count'])
y_test = test_data['Burglary count']

# Define the objective function for Optuna using sklearn.cross_val_score
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0)
    }
    model = XGBRegressor(objective='reg:squarederror', random_state=42, **params)
    # Compute CV scores using sklearn's cross_val_score
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
    return -cv_scores.mean()

# Create and run the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("Best hyperparameters:", study.best_params)
print("Best CV MSE:", study.best_value)

# Train the final model using the best parameters from Optuna
best_model = XGBRegressor(objective='reg:squarederror', random_state=42, **study.best_params)
best_model.fit(X_train, y_train)

# Make predictions on the test set.
y_pred = best_model.predict(X_test)

# Compute MSE.
mse = mean_squared_error(y_test, y_pred)
r2_score = r2_score(y_test, y_pred)
print(f"Test MSE: {mse}\nTest R^2: {r2_score}")

# SHAP Analysis
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

plt.rcParams.update({'font.size': 14})
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)

# Add predictions to the test data and save to CSV
test_data['Predicted Incident Count'] = y_pred
test_data.to_csv('all_data_predictions_2024_2025.csv', index=False)

# Save the best model
best_model.save_model('all_data_xgboost_model.json')