"""Plain-language presentation, without modifying numerical outputs."""

import pandas as pd

LIFE_HELP = "The model's estimate of how many operating cycles remain."
RANGE_HELP = "A development-calibrated uncertainty range. It is not a failure probability or guarantee."
ATTENTION_HELP = "Triggered when the lower end of the exploratory uncertainty range is at or below 30 cycles."


def attention_label(flag):
    return "Attention" if flag else "No attention flag"


def range_label(lower, upper):
    return f"{lower:.1f}–{upper:.1f} cycles"


def priority_table(results):
    return pd.DataFrame({
        "Engine": [f"Engine {int(i):03d}" for i in results.unit_number],
        "Current cycle": results.time_in_cycles.to_numpy(),
        "Estimated life": [f"{v:.1f} cycles" for v in results.predicted_rul],
        "Exploratory range": [range_label(lo, hi) for lo, hi in zip(results.lower_bound, results.upper_bound)],
        "Status": [attention_label(flag) for flag in results.low_rul_flag],
    })
