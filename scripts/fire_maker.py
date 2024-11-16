import logging
import multiprocessing
import os

import netCDF4 as nc
import numpy as np
import pandas as pd


def process_month(group, year, month, variables, root):
    """
    Process data for a specific year and month group.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing data for Year: {year}, Month: {month}")

    columns = []
    variable_levels = {}
    datasets = {}

    for var in variables:
        nc_file_path = f"{root}/{var}/{var}.{year}{str(month).zfill(2)}.nc"
        if not os.path.exists(nc_file_path):
            logger.warning(
                f"File {nc_file_path} does not exist. Skipping variable {var}.")
            continue

        logger.info(f"Opening netCDF file for variable {var}: {nc_file_path}")
        ds = nc.Dataset(nc_file_path, 'r')
        datasets[var] = ds

        var_data = ds.variables[var]
        dims = var_data.dimensions

        if 'level' in dims:

            levels = ds.variables['level'][:]
            level_cols = [f"{var}_level_{int(level)}" for level in levels]
            columns.extend(level_cols)
            variable_levels[var] = levels
        elif 'time' in dims and 'y' in dims and 'x' in dims:

            columns.append(var)
        else:
            logger.warning(f"Variable {var} has unexpected dimensions: {dims}")

            pass

    columns = list(set(columns))

    data_dicts = []

    if datasets:
        any_ds = next(iter(datasets.values()))
        x = any_ds.variables['x'][:]
        y = any_ds.variables['y'][:]
        times = any_ds.variables['time'][:]
        time_units = any_ds.variables['time'].units
        time_calendar = getattr(
            any_ds.variables['time'], 'calendar', 'standard')
        dates_nc = nc.num2date(times, units=time_units, calendar=time_calendar)

        x_coords = group['longitude'].unique()
        y_coords = group['latitude'].unique()
        x_coord_to_idx = {x_coord: np.argmin(
            np.abs(x - x_coord)) for x_coord in x_coords}
        y_coord_to_idx = {y_coord: np.argmin(
            np.abs(y - y_coord)) for y_coord in y_coords}

        dates_in_group = group['initialdate'].unique()
        date_indices = {}
        for date in dates_in_group:

            time_diffs = np.abs([(dt - date).total_seconds()
                                for dt in dates_nc])
            time_idx = np.argmin(time_diffs)

            if time_diffs[time_idx] > (12 * 3600):
                logger.warning(f"No time close to {date} in datasets")
                continue
            date_indices[date] = time_idx
    else:
        logger.warning(
            f"No datasets available for Year: {year}, Month: {month}")
        return

    for idx, row in group.iterrows():
        row_dict = row.to_dict()
        date = row['initialdate']
        x_coord = row['longitude']
        y_coord = row['latitude']

        time_idx = date_indices.get(date)
        x_idx = x_coord_to_idx.get(x_coord)
        y_idx = y_coord_to_idx.get(y_coord)
        if time_idx is None or x_idx is None or y_idx is None:
            logger.warning(
                f"Indices not found for idx {idx}: date {date}, x {x_coord}, y {y_coord}")
            continue

        for var, ds in datasets.items():
            var_data = ds.variables[var]
            dims = var_data.dimensions
            try:
                if 'level' in dims:

                    data_values = var_data[time_idx, :,
                                           y_idx, x_idx]
                    levels = variable_levels[var]
                    for level_idx, level_value in enumerate(levels):
                        data_value = data_values[level_idx]
                        col_name = f"{var}_level_{int(level_value)}"
                        row_dict[col_name] = data_value.item()
                elif 'time' in dims and 'y' in dims and 'x' in dims:
                    data_value = var_data[time_idx, y_idx, x_idx]
                    row_dict[var] = data_value.item()
                else:

                    pass
            except IndexError as e:
                logger.warning(
                    f"IndexError for variable {var} at idx {idx}: {e}")
                if 'level' in dims:
                    levels = variable_levels[var]
                    for level_value in levels:
                        col_name = f"{var}_level_{int(level_value)}"
                        row_dict[col_name] = np.nan
                else:
                    row_dict[var] = np.nan

        data_dicts.append(row_dict)

    for ds in datasets.values():
        ds.close()

    group_results_df = pd.DataFrame(data_dicts)

    os.makedirs("./fires", exist_ok=True)

    output_file = f"./fires/fire_{year}{str(month).zfill(2)}.csv"

    logger.info(
        f"\033[32mSaving results for Year: {year}, Month: {month} to {output_file}\033[0m")
    group_results_df.to_csv(output_file, index=False)


def main():

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    root = "/media/bharxhav/valhalla/NARR"
    variables = os.listdir(root)

    df = pd.read_csv('./assets/fire_locs.csv')

    df['initialdate'] = pd.to_datetime(df['initialdate'])
    df['year'] = df['initialdate'].dt.year
    df['month'] = df['initialdate'].dt.month

    grouped = df.groupby(['year', 'month'])

    args_list = [(group, year, month, variables, root)
                 for (year, month), group in grouped]

    cpu_count = multiprocessing.cpu_count()
    logger.info(f"Starting multiprocessing with {cpu_count} processes")
    with multiprocessing.Pool(processes=cpu_count) as pool:
        pool.starmap(process_month, args_list)

    logger.info("Processing completed.")


if __name__ == "__main__":
    main()
