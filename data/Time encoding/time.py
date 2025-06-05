import numpy as np
import pandas as pd

baseline = pd.read_csv('C:/Users/20231441/OneDrive - TU Eindhoven/Documents/GitHub/MD-CBL-Group-12/data/Base/baseline_dataset.csv')
baseline.drop(columns=['LSOA code 2011', 'LSOA name 2021', 'Change Indicator'], inplace=True, errors='ignore')

baseline["Month encoding (sin)"] = np.sin(2 * np.pi * baseline["Month"] / 12).round(4)
baseline["Month encoding (cos)"] = np.cos(2 * np.pi * baseline["Month"] / 12).round(4)

baseline.to_csv('time_encoding_finalized.csv', index=False)
print(baseline.head(20))