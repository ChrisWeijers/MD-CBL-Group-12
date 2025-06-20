import numpy as np
import pandas as pd
from pathlib import Path

def main():
    data_dir = Path(__file__).resolve().parent.parent
    baseline_file = data_dir / 'Base/baseline_dataset.csv'
    baseline = pd.read_csv(baseline_file)
    baseline.drop(columns=['LSOA code 2011', 'LSOA name 2021', 'Change Indicator'], inplace=True, errors='ignore')

    baseline["Month encoding (sin)"] = np.sin(2 * np.pi * baseline["Month"] / 12).round(4)
    baseline["Month encoding (cos)"] = np.cos(2 * np.pi * baseline["Month"] / 12).round(4)

    baseline.to_csv(data_dir / 'Time_encoding/time_encoding_finalized.csv', index=False)
    print(baseline.head(20))