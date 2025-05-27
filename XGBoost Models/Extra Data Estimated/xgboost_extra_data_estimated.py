import optuna
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score  # using the sklearn function
import matplotlib.pyplot as plt
import numpy as np
import shap
from pathlib import Path

# Load the aggregated data
base_dir = Path(__file__).resolve().parent.parent.parent
data_file = base_dir / 'data/burglary_london.csv'
df = pd.read_csv(data_file)
df[['Year', 'Month']] = df['Month'].str.split('-', n=1, expand=True)
data = df.groupby(['Year', 'Month', 'LSOA code']).size().reset_index(name='Burglary')

# Load the extra data with estimations
extra_data_file = base_dir / 'data/extra_data.csv'
extra_data = pd.read_csv(extra_data_file)
extra_data.rename(columns={'year': 'Year', 'month': 'Month'}, inplace=True)
extra_data = extra_data[~((extra_data['Year'] == 2025) & (extra_data['Month'] > 2))]

# Ensure consistent data types for merging keys
extra_data['Year'] = extra_data['Year'].astype(int)
extra_data['Month'] = extra_data['Month'].astype(int)
data['Year'] = data['Year'].astype(int)
data['Month'] = data['Month'].astype(int)

# Merge the grid with the actual burglary data
full_data = pd.merge(extra_data, data, on=['Year','Month','LSOA code'], how='left')
full_data.fillna(0, inplace=True)

# Encode categorical variable for LSOA code while preserving the original
full_data['LSOA code encoded'] = full_data['LSOA code'].astype('category').cat.codes

# Split the data into training (2010-2023) and testing (2024-2025) sets
train_data = full_data[full_data['Year'] <= 2023].copy()
test_data = full_data[full_data['Year'] >= 2024].copy()

# Prepare features (X) and target (y) for training and testing
X_train = train_data.drop(columns=['Year', 'Month', 'LSOA code', 'Burglary'])
y_train = train_data['Burglary']
X_test = test_data.drop(columns=['Year', 'Month', 'LSOA code', 'Burglary'])
y_test = test_data['Burglary']

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
study.optimize(objective, n_trials=15, show_progress_bar=True)

print("Best hyperparameters:", study.best_params)
print("Best CV MSE:", study.best_value)

# Train the final model using the best parameters from Optuna
best_model = XGBRegressor(objective='reg:squarederror', random_state=42, **study.best_params)
best_model.fit(X_train, y_train)

# Make predictions on the test set.
y_pred = best_model.predict(X_test)

# Compute MSE.
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {mse}\nTest R2: {r2}")

# SHAP Analysis
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

plt.rcParams.update({'font.size': 14})
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)

# Add predictions to the test data and save to CSV
test_data['Predicted Incident Count'] = y_pred
test_data.to_csv('extra_data_estimation_predictions_2024_2025.csv', index=False)

# Save the best model
best_model.save_model('extra_data_estimation_xgboost_model.json')