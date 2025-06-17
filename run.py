import sys
import subprocess

print('Downloading data from google drive...')
import data_download
print('Data downloaded successfully!')
print('===================================================================')
print('Loading data files...')
import data.LSOA_changes.lsoafilter
import data.Base.basedataset
import data.Crimes.crimes
import data.Covid_19.covid
import data.Burglary_lag.burglary_lag
import data.Daylight.daylight
import data.IMD.imd
import data.Education.education
import data.Holidays_and_celebrations.holidays_and_celebrations
import data.Hours_worked.hours_worked
import data.Household_income.household_income
import data.Land_use.land
import data.Population.population
import data.Precipitation.precipitation
import data.Standard_Deviation.std
import data.Time_encoding.time
print('Data loaded successfully!')
print('===================================================================')
print('===Running XGBoost model...===')
import XGBoost_Models.all_data_xgboost
print('===XGBoost model run successfully!===')
print('===================================================================')
print('Preparing dashboard...')
import app.Dashboard_GeoJSONCombine
print('Running dashboard...')
subprocess.run([sys.executable, 'app/DashLeafletNew.py'])
print('Dashboard run successfully!')
print('===================================================================')

