"""Create leakage-safe dataset splits"""

# If we do not split by engine first, have data leakage
# The normal train_test_split splits dataset row by row

# But in our dataset, 1 engine can appear in many rows with different time in cycle
# So normal train_test_split won't work because
# cycles from engine could appear in both sets:

# Training: Engine 1: cycles 1-150
# Testing: Engine 1: cycles 151-192
# => The model has already seen => BAD

import pandas as pd

from sklearn.model_selection import train_test_split

def split_by_engine(
        data: pd.DataFrame,
        validation_size: float = 0.2,
        random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split complete engine trajectories into training and validation sets."""


    # get the engine unit number + drop duplicate
    # + turn into numpy array
    engine_ids = data["unit_number"].drop_duplicates().to_numpy()

    # for the unique engine ids array
    # split 80% into training and 20% into validation
    training_ids, validation_ids = train_test_split(
        engine_ids,
        random_state=random_state,
        test_size=validation_size,
        shuffle=True
    )

# create a boolean mask to collect all engines used for training
    training_data = data[
        data["unit_number"].isin(training_ids)
    ].copy()

# create a boolean mask to collect all engine used for validation
    validation_data = data[
        data["unit_number"].isin(validation_ids)
    ].copy()

    return training_data, validation_data