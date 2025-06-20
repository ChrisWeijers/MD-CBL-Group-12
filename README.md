# 🚓 Police Force Allocation Prediction Model for the Metropolitan Police

This project aims to predict the amount of police officers needed in each LSOA in London, to best reduce the amount of burglaries. The model first predicts the number of burglaries expected each month in every LSOA across London. Based on these predictions, it calculates the number of policing hours required in each area to minimize burglary rates effectively.


## Getting Started

1. **Clone the repository**:

Navigate to your desired directory and run:
```bash
git clone https://github.com/ChrisWeijers/MD-CBL-Group-12.git
```
2. **Install dependencies**:

Navigate into the project directory and install all required Python packages:
```bash
pip install -r requirements.txt
```
3. **Download Police Data**:

We used crime data from [data.police.uk](https://data.police.uk). Since the specific dataset used in this project is not directly downloadable in a single click from the website, we've made it available via Google Drive for convenience. 

[Download the data from Google Drive](https://drive.google.com/drive/folders/1_W5TvYWFbiOBpO7Yk_TZkhqYYyT6Q00b?usp=share_link) 

After downloading, place the file in the following directory, and ensist it has the name 'Original Dataset':
```bash
data/Crimes/
```
4. **Download Land Use Data**:

The files used for the 'Land Use' data from [Geofabrik](https://download.geofabrik.de/europe/united-kingdom/england/greater-london.html) are too big for GitHub and need to be downloaded.

[Download the data from Geofabrik](https://download.geofabrik.de/europe/united-kingdom/england/greater-london-latest-free.shp.zip) 

After downloading, place the file in the following directory, and ensist it has the name 'Original Dataset':
```bash
data/Land_use/
```
You are now fully set to run the project!
## Running the project

To run all components of the project, including data gathering, processing, cleaning, formatting, estimation, model training, and dashboard creation, execute the following command in your terminal:
```bash
python run.py
```
This will take roughly 30-45 minutes, and automatically opens the dashboard. 

## Scripts Overview

### 1. Data Collection
- `data_download.py`: Downloads raw datasets from Google Drive.

### 2. 📁data: Enriching with External Data
All data is preprocessed, cleaned, formatted and estimated in the `data/` directory. Each folder loads and prepares different data. 

- `Base/`: Baseline data to be merged with all other datasets
- `Burglary_lag/`: Lagged burglary counts, EMAs and SMAs
- `Covid_19/`: Covid-19 boolean
- `Crimes/`: Crime data
- `Daylight/`: Daylight duration data
- `Education/`: Education level data
- `Holidays_and_celebrations/`: Public holidays and major event dates
- `Hours_worked/`: Average working hours data
- `Household_income/`: Household income data
- `IMD/`: Index of Multiple Deprivation (socioeconomic data)
- `Land_use/`: Geographic and land usage data
- `LSOA_boundaries/`: Shapefiles of LSOAs
- `LSOA_changes/`: Historical changes to LSOA definitions
- `Population/`: Total population data
- `Population_density/`: Population density data
- `Precipitation/`: Rainfall data
- `Smoothed burglaries/`: Smoothed burglary time series
- `Standard_Deviation/`: Standard deviation of burglary counts over time
- `Time_encoding/`: sine/cosine month encoding

### 3. 📁XGBoost_models: Burglary Prediction Models
- `all_data_xgboost.py`: Trains model and predicts monthly burglary count.
- `all_data_xgboost_test.py`: Trains and test the model.

### 4. 📁app: Dashboard Preparation
- `Dashboard_GeoJSONCombine.py`: Merges shapefiles into a single GeoJSON for mapping.
- `DashLeafletNew.py`: Generates an interactive dashboard displaying burglary count predictions and policing hours distribution by LSOA or ward.

### 5. 📁visualizations: Burglary Visualizations
- This directory contains some files which create visualizations. These visualizations are used for the report and presentations. 

