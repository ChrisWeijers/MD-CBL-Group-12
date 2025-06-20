import sys
import subprocess

print('Downloading data from google drive...')
import data_download as download
download.main()
print('Data downloaded successfully!')
print('===================================================================')

import data.LSOA_changes.lsoafilter as lsoa
print('Preparing LSOA changes...')
lsoa.main()
import data.Base.basedataset as base
base.main()
import data.Crimes.crimes as crimes
print('Preparing crimes...')
crimes.main()
import data.Covid_19.covid as covid
print('Preparing Covid-19 boolean...')
covid.main()
import data.Burglary_lag.burglary_lag as burglary
print('Preparing Burglary lag...')
burglary.main()
import data.Daylight.daylight as daylight
print('Preparing Daylight...')
daylight.main()
import data.IMD.imd as imd
print('Preparing IMD...')
imd.main()
import data.Education.education as education
print('Preparing Education...')
education.main()
import data.Holidays_and_celebrations.holidays_and_celebrations as holidays
print('Preparing Holidays and celebrations...')
holidays.main()
import data.Hours_worked.hours_worked as hours
print('Preparing Hours worked...')
hours.main()
import data.Household_income.household_income as income
print('Preparing Household income...')
income.main()
import data.Land_use.land as land
print('Preparing Land use...')
land.main()
import data.Population.population as population
print('Preparing Population...')
population.main()
import data.Population_density.population_density as population_density
print('Preparing Population density...')
population_density.main()
import data.Precipitation.precipitation as precipitation
print('Preparing Precipitation...')
precipitation.main()
import data.Standard_Deviation.std as std
print('Preparing standard deviation...')
std.main()
import data.Time_encoding.time as time
print('Preparing time encodings...')
time.main()
print('Data loaded successfully!')
print('===================================================================')
print('===Running XGBoost model...===')
import XGBoost_Models.all_data_xgboost as model
model.main()
print('===XGBoost model run successfully!===')
print('===================================================================')
print('Preparing dashboard...')
import app.Dashboard_GeoJSONCombine as combine
combine.main()
print('Running dashboard...')
subprocess.run([sys.executable, 'app/DashLeafletNew.py'])
print('Dashboard run successfully!')
print('===================================================================')

