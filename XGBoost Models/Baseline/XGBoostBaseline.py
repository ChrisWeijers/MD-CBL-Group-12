import pandas as pd
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import shap
import matplotlib.pyplot as plt

# Load the primary aggregated data CSV
data_file = 'C:/Users/20231441/Desktop/University/Year 2/Q4/CBL - Crime/Combined & Cleaned Dataset/Combined Dataset/All Crimes London Aggregated (2010 - 2025).csv'
data = pd.read_csv(data_file)

# Filter for burglary cases in the data
burglary_data = data[data['Crime type'] == 'Burglary'].copy()

# Load the third CSV that I'm using only to get all of the LSOA codes
lsoa_file = 'C:/Users/20231441/Desktop/University/Year 2/Q4/CBL - Crime/lsoa-data.csv'
lsoa_data = pd.read_csv(lsoa_file, header=2)
lsoa_data = lsoa_data[['Codes']]
lsoa_data.rename(columns={'Codes': 'LSOA code'}, inplace=True)

# Make a dataframe "grid" with every year, month, and LSOA code combination
unique_year_month = data[['Year', 'Month']].drop_duplicates()
unique_lsoa = pd.DataFrame(lsoa_data['LSOA code'].unique(), columns=['LSOA code'])
grid = unique_year_month.merge(unique_lsoa, how='cross')

# Merge the grid with the actual burglary data
full_data = pd.merge(grid, burglary_data, on=['Year','Month','LSOA code'], how='left')
full_data['Incident Count'] = full_data['Incident Count'].fillna(0)

# Encode categorical variable for LSOA code while preserving the original
full_data['LSOA code encoded'] = full_data['LSOA code'].astype('category').cat.codes

# Split the data into training (2010-2023) and testing (2024-2025) sets
train_data = full_data[full_data['Year'] <= 2023].copy()
test_data = full_data[full_data['Year'] >= 2024].copy()

# Prepare features (X) and target (y)
# Baseline features: only 'Month' and 'LSOA code encoded'
X_train = train_data[['Month', 'LSOA code encoded']]
y_train = train_data['Incident Count']
X_test = test_data[['Month', 'LSOA code encoded']]
y_test = test_data['Incident Count']

# Define the objective function for Optuna with baseline parameters
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
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error').mean()
    return -score

# Create and run the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("Best hyperparameters:", study.best_params)
print("Best CV MSE:", study.best_value)

# Train the final model using the best parameters from Optuna
best_model = XGBRegressor(objective='reg:squarederror', random_state=42, **study.best_params)
best_model.fit(X_train, y_train)

# Make predictions on the test set and compute MSE
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse}")

# SHAP Analysis
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

plt.rcParams.update({'font.size': 14})
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)

# Add predictions to the test data and save to CSV
test_data['Predicted Incident Count'] = y_pred
test_data['Crime type'].fillna('Burglary', inplace=True)
test_data.to_csv('burglary_predictions_2024_2025.csv', index=False)

# Save the best model
best_model.save_model('burglary_xgboost_model.json')