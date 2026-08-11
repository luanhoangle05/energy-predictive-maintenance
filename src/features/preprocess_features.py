"""Create leakage-safe feature PREPROCESSING components."""

from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Goal: remove sensor with no variance/changing

# Ex: sensor 1 -> 10,20,30,40  changes a lot
#     sensor 2 -> 50,50,50,50  never changes => variance = 0
#     sensor 3 -> 1,1,1,1      never changes => variance = 0

# So VarianceThreshold(threshold=0.0) removes sensor whose variance = 0
# This would keep only sensor 1

def create_feature_preprocessor(
        variance_threshold: float = 0.0
) -> Pipeline:
    """Create an unfitted pipeline that removes low-variance features."""

    # creates a sequence of preprocessing steps
    preprocessor = Pipeline(
        steps=[
            (
                "variance_filter",
                VarianceThreshold(
                    threshold=variance_threshold
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            )
        ]
    )

    return preprocessor.set_output(transform="pandas")
