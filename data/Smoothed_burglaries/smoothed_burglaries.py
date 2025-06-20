import os
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from tqdm import tqdm
from pathlib import Path

def main():
    # Load the necessary data files
    data_dir = Path(__file__).resolve().parent.parent
    baseline_csv = data_dir / 'Base/baseline_dataset.csv'
    shp_folder = data_dir / 'LSOA_boundaries/LB_shp'
    crimes_csv = data_dir / 'Crimes/crimes_finalized.csv'

    # Set the data for plotting
    plot_date = pd.to_datetime("2023-06-01")

    # Load all LSOA shapefiles and build adjacencies
    # Read every .shp file in shp_folder
    shp_files = glob.glob(os.path.join(shp_folder, "*.shp"))
    gdf_list = [gpd.read_file(f) for f in shp_files]

    if len(gdf_list) == 0:
        raise ValueError(f"No shapefiles found under {shp_folder!r}. Make sure the path is correct.")

    # Concatenate them into one GeoDataFrame
    gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))

    # Rename the LSOA column so it matches your crime/baseline CSVs
    gdf = gdf.rename(columns={"lsoa21cd": "LSOA code 2021"})

    # Reproject to British National Grid (EPSG:27700)
    if gdf.crs is None or gdf.crs.to_string() != "EPSG:27700":
        gdf = gdf.to_crs("EPSG:27700")

    # Add a “geom_index” column that we will use as a numeric index for adjacency
    gdf = gdf.reset_index().rename(columns={"index": "geom_index"})
    gdf = gdf[["geom_index", "LSOA code 2021", "geometry"]]

    # Build an adjacency dict: for each geom_index, list all neighbouring geom_index (plus itself)
    sindex = gdf.sindex
    neighbor_dict = {}

    print("Building adjacency list (LSOA polygons)…")
    for idx, geom in tqdm(gdf.geometry.items(), total=len(gdf)):
        candidates = list(sindex.intersection(geom.bounds))
        neighbors = [cand for cand in candidates if cand != idx and geom.touches(gdf.geometry.iloc[cand])]
        neighbors.append(idx)
        neighbor_dict[idx] = neighbors

    # Lookups between LSOA code and geom_index
    code_to_index = dict(zip(gdf["LSOA code 2021"], gdf["geom_index"]))
    index_to_code = dict(zip(gdf["geom_index"], gdf["LSOA code 2021"]))

    # Load and pivot crime counts into a monthly matrix
    crimes = pd.read_csv(crimes_csv, usecols=["LSOA code 2021", "Year", "Month", "Burglary count"])
    if crimes.empty:
        raise ValueError(f"No burglary data found in {crimes_csv!r}.")
    crimes["Year"] = crimes["Year"].astype(int)
    crimes["Month"] = crimes["Month"].astype(int)

    # Build a true datetime column
    crimes["date"] = pd.to_datetime(crimes[["Year", "Month"]].assign(DAY=1))

    # Filter to the period 2011-01 through 2025-02
    crimes = crimes[(crimes["date"] >= "2011-01-01") & (crimes["date"] <= "2025-04-01")]

    # Pivot the data
    pivot_df = crimes.pivot_table(
        index="date",
        columns="LSOA code 2021",
        values="Burglary count",
        aggfunc="sum",
        fill_value=0,
    )

    # Ensure every calendar month is present as a row
    all_months = pd.date_range(start="2011-01-01", end="2025-04-01", freq="MS")
    pivot_df = pivot_df.reindex(all_months, fill_value=0)

    # Ensure every LSOA code (from shapefile) is a column
    all_codes = gdf["LSOA code 2021"].unique().tolist()
    pivot_df = pivot_df.reindex(columns=all_codes, fill_value=0)

    # Compute a 12-month rolling sum of burglary counts, excluding the current month
    shifted = pivot_df.shift(1).fillna(0)

    # Take a rolling 12-month window on that shifted series
    rolling_sum = shifted.rolling(window=12, min_periods=1).sum()

    # For each month, smooth the windowed sum by adjacency
    smoothed_all = pd.DataFrame(index=rolling_sum.index, columns=rolling_sum.columns, dtype=float)

    print("Performing adjacency smoothing for every month…")
    for date in tqdm(rolling_sum.index, desc="Smoothing over dates"):
        row = rolling_sum.loc[date]
        smoothed_vals = {}
        for code in row.index:
            idx = code_to_index.get(code)
            if idx is None:
                smoothed_vals[code] = 0.0
                continue
            neigh_idxs = neighbor_dict.get(idx, [idx])
            neigh_codes = [index_to_code[nidx] for nidx in neigh_idxs]
            neigh_vals = row[neigh_codes].values
            smoothed_vals[code] = neigh_vals.mean()
        smoothed_all.loc[date] = pd.Series(smoothed_vals)

    # Convert to "expected monthly burglaries"
    expected_12_month = pd.DataFrame(index=smoothed_all.index, columns=smoothed_all.columns, dtype=float)
    expected_6_month = pd.DataFrame(index=smoothed_all.index, columns=smoothed_all.columns, dtype=float)
    expected_3_month = pd.DataFrame(index=smoothed_all.index, columns=smoothed_all.columns, dtype=float)

    for i, date in enumerate(smoothed_all.index):
        window_len_12 = min(i, 12)
        window_len_6 = min(i, 6)
        window_len_3 = min(i, 3)

        if window_len_12 == 0:
            expected_12_month.loc[date] = 0.0
        else:
            expected_12_month.loc[date] = smoothed_all.loc[date] / window_len_12

        if window_len_6 == 0:
            expected_6_month.loc[date] = 0.0
        else:
            expected_6_month.loc[date] = smoothed_all.loc[date] / window_len_6

        if window_len_3 == 0:
            expected_3_month.loc[date] = 0.0
        else:
            expected_3_month.loc[date] = smoothed_all.loc[date] / window_len_3

    # Explode into long form and merge with baseline grid
    # Expected 12-month values
    exp_long_12 = (
        expected_12_month.reset_index()
        .melt(id_vars="index", var_name="LSOA code 2021", value_name="Smoothed_burglaries (12 months)")
        .rename(columns={"index": "date"})
    )
    exp_long_12["Year"] = exp_long_12["date"].dt.year
    exp_long_12["Month"] = exp_long_12["date"].dt.month

    # Expected 6-month values
    exp_long_6 = (
        expected_6_month.reset_index()
        .melt(id_vars="index", var_name="LSOA code 2021", value_name="Smoothed_burglaries (6 months)")
        .rename(columns={"index": "date"})
    )
    exp_long_6["Year"] = exp_long_6["date"].dt.year
    exp_long_6["Month"] = exp_long_6["date"].dt.month

    # Expected 3-month values
    exp_long_3 = (
        expected_3_month.reset_index()
        .melt(id_vars="index", var_name="LSOA code 2021", value_name="Smoothed_burglaries (3 months)")
        .rename(columns={"index": "date"})
    )
    exp_long_3["Year"] = exp_long_3["date"].dt.year
    exp_long_3["Month"] = exp_long_3["date"].dt.month

    # --- Merge the long DataFrames into one ---
    # Merge exp_long_12 with the exp_long_6 (on date and LSOA code 2021)
    exp_long_all = exp_long_12.merge(
        exp_long_6[["date", "LSOA code 2021", "Smoothed_burglaries (6 months)"]],
        on=["date", "LSOA code 2021"],
        how="left",
    )

    # Merge the result with exp_long_3
    exp_long_all = exp_long_all.merge(
        exp_long_3[["date", "LSOA code 2021", "Smoothed_burglaries (3 months)"]],
        on=["date", "LSOA code 2021"],
        how="left",
    )

    # Recompute Year and Month, if needed
    exp_long_all["Year"] = exp_long_all["date"].dt.year
    exp_long_all["Month"] = exp_long_all["date"].dt.month

    # Merge with the baseline grid
    baseline = pd.read_csv(baseline_csv, dtype={"LSOA code 2021": str, "Year": int, "Month": int})
    baseline = baseline[["LSOA code 2021", "Year", "Month"]]

    merged = baseline.merge(
        exp_long_all[
            ["LSOA code 2021", "Year", "Month", "Smoothed_burglaries (12 months)",
             "Smoothed_burglaries (6 months)", "Smoothed_burglaries (3 months)"]
        ],
        on=["LSOA code 2021", "Year", "Month"],
        how="left",
    )

    # If Year = 2011 and Month = 1 set the smoothed columns to NaN
    mask = (merged["Year"] == 2011) & (merged["Month"] == 1)
    cols = ["Smoothed_burglaries (12 months)", "Smoothed_burglaries (6 months)", "Smoothed_burglaries (3 months)"]
    merged.loc[mask, cols] = np.nan

    # Round the smoothed expected values to 2 decimal points
    merged["Smoothed_burglaries (12 months)"] = merged["Smoothed_burglaries (12 months)"].round(2)
    merged["Smoothed_burglaries (6 months)"] = merged["Smoothed_burglaries (6 months)"].round(2)
    merged["Smoothed_burglaries (3 months)"] = merged["Smoothed_burglaries (3 months)"].round(2)

    # Save the final merged table
    merged.reset_index(drop=True, inplace=True)
    merged.to_csv(data_dir / "Smoothed_burglaries/smoothed_burglaries_finalized.csv", index=False)

    # 2D choropleth plot
    plot_df_2d = exp_long_all[exp_long_all["date"] == plot_date][["LSOA code 2021", "Smoothed_burglaries (12 months)"]]
    plot_gdf_2d = gdf.merge(plot_df_2d, on="LSOA code 2021", how="left")
    plot_gdf_2d["Smoothed_burglaries (12 months)"] = plot_gdf_2d["Smoothed_burglaries (12 months)"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_gdf_2d.plot(
        column="Smoothed_burglaries (12 months)",
        ax=ax,
        cmap="OrRd",
        legend=True,
        missing_kwds={"color": "lightgrey", "edgecolor": "red", "hatch": "///"}
    )
    ax.set_title(f"2D: Smoothed Monthly Burglaries for {plot_date.strftime('%Y-%m')}")
    ax.set_axis_off()

    # Save the 2D plot
    plt.savefig("2d smoothed expected burglaries.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # 3D surface plot
    plot_gdf_2d["cen_x"] = plot_gdf_2d.geometry.centroid.x
    plot_gdf_2d["cen_y"] = plot_gdf_2d.geometry.centroid.y

    x = plot_gdf_2d["cen_x"].values
    y = plot_gdf_2d["cen_y"].values
    z = plot_gdf_2d["Smoothed_burglaries (12 months)"].values

    triang = mtri.Triangulation(x, y)

    fig = plt.figure(figsize=(12, 9))
    ax3 = fig.add_subplot(111, projection="3d")
    surf = ax3.plot_trisurf(triang, z, cmap="viridis", linewidth=0.2, antialiased=True)
    ax3.set_xlabel("Easting (m)", labelpad=10)
    ax3.set_ylabel("Northing (m)", labelpad=10)
    ax3.set_zlabel("Smoothed Monthly Burglaries", labelpad=10)
    ax3.set_title(f"3D: Smoothed Monthly Burglaries for {plot_date.strftime('%Y-%m')}")
    fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=10, pad=0.1, label="Burglaries/month")

    # Save the 3D plot
    plt.savefig("3d smoothed expected burglaries.png", bbox_inches="tight", dpi=150)
    plt.close(fig)