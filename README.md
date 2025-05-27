# 🚓 Police Force Allocation Prediction Model for the Metropolitan Police

This project aims to predict the amount of police officers needed in each LSOA in London, to best reduce the amount of burglaries. The model first predicts the number of burglaries expected each month in every LSOA across London. Based on these predictions, it estimates the number of police officers required in each area to minimize burglary rates effectively.


## Getting Started

1. **Clone the repository**:
```bash
git clone https://github.com/ChrisWeijers/MD-CBL-Group-12.git
```
2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Scripts Overview

Run these scripts in the order they appear to get the desired outcome. 

### 1. Data Collection and Processing
- `data_download.py`: Downloads raw datasets from Google Drive.
- `process_data.py`: Cleans police data and assigns ward code and name to each row using LSOA lookups.

### 2. Enriching with External Data
- `extra_data_no_estimates.py`: Loads additional datasets, namely:
    - population density,
    - average hours of worked in a week,
    - qualifications (education level),
    - Index of Multiple Deprivation (IMD),
    - average number of daylight hours per day for each month.
- `extra_data.py`: Linearly interpolates missing months in external data.

### 3. Burglary Prediction Models
- `xgboost_extra_data_no_estimation.py`: Predicts monthly burglary count using only available (non-estimated) data.
- `xgboost_extra_data_estimated.py`: Uses interpolated data to train a more complete model.

### 4. Dashboard Preparation
- `Dashboard_GeoJSONCombine.py`: Merges shapefiles into a single GeoJSON for mapping.
- `Dashboard_HTML.py`: Generates an interactive dashboard displaying predictions and officer allocation by LSOA, clicking LSOA gives more information.

