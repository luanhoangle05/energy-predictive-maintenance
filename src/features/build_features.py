"""Build model-ready features from validated sensor data."""

import pandas as pd

from src.data.load_data import RUL_COLUMN

NON_FEATURE_COLUMNS = [
    "unit_number",
    RUL_COLUMN
]

def add_rolling_sensor_means(
        data: pd.DataFrame,
        window_size: int = 5,
) -> pd.DataFrame:
    """Add rolling sensor means calculated independently per engine."""
    result = data.copy()

    # collecting sensor 1 - sensor 21
    sensor_columns = [
        column
        for column in result.columns
        if column.startswith("sensor_")
        and "_rolling_" not in column
    ]

# window_size = 5
# means: current cycle and 4 previous cycles

    for sensor_column in sensor_columns:
        feature_name = (
            f"{sensor_column}_rolling_mean_{window_size}"
        )

        result[feature_name] = result.groupby(
            "unit_number",
            sort= False,
        )[sensor_column].transform(
            lambda values: values.rolling(
                window = window_size,
                min_periods=1
            ).mean()
        )

    return result

# group by engine first
# because we run the for loop through 21 sensor column
# so it will take the values belong to that sensor

# Example we have data
# unit number    cycle    sensor 1    sensor 2   sensor 3
#     1            1         10          20          30
#     1            2         12           22         32
#     1            3         14           24         34
#     2            1         100         200         300
#     2            2         110         210         310

# So the interation result.groupby("unit_number")[sensor_column]
# will effectively look like: result.groupby("unit_number")[sensor_1]
# so engine 1 sensor 1: 10, 12, 14 => Then calculate the mean
# engine 1 sensor 2: 20,22,24



# Goal: if 2 engines have the same mean
# Standard Deviation helps detect unstable or noisier change
def add_rolling_sensor_standard_deviations(
        data: pd.DataFrame,
        window_size: int = 5,
) -> pd.DataFrame:
    """Add rolling sensor variability calculated independently per engine."""
    result = data.copy()

# we have to add codition: "_rolling_" not in column
# because after function: add_rolling_sensor_means
# we have sensor_1 and sensor_1_rolling_mean_5
# but we want to make sure we use the original to calculate std

    sensor_columns = [
        column
        for column in result.columns
        if (
            column.startswith("sensor_")
            and "_rolling_" not in column
        )
    ]

# ddof = 0 avoids producing missing value at first cycle
# Ex: at cycle 1, there is only [50] so std will be =0 instead of Nan

    for sensor_column in sensor_columns:
        feature_name = (
            f"{sensor_column}_rolling_std_{window_size}"
        )

        result[feature_name] = result.groupby(
            "unit_number",
            sort=False,
        )[sensor_column].transform(
            lambda values: values.rolling(
            window=window_size,
            min_periods=1,
        ).std(ddof=0)
        )
    return  result



def build_features(data: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:
    """Return model inputs without engine identifiers or RUL targets."""

    featured_data = add_rolling_sensor_means(
        data,
        window_size= rolling_window,
    )

    featured_data = add_rolling_sensor_standard_deviations(
        featured_data,
        window_size=rolling_window,
    )

    columns_to_remove = []

    for column in NON_FEATURE_COLUMNS:
        if column in featured_data.columns:
            columns_to_remove.append(column)

    return featured_data.drop(columns=columns_to_remove).copy()
