"""fixation.py – Fixation analysis compatible with the rescue-grid dataset.

Wraps detectors.fixation_detection and accepts either:
  - numpy arrays (x, y, time)  — same interface as detectors.py
  - a pandas DataFrame          — auto-extracts columns using config

Usage
-----
From arrays:
    from igaze.fixation import extract_fixations
    Sfix, Efix = extract_fixations(x, y, time)

From a DataFrame (eyetracker parquet/csv saved by xdf.py):
    from igaze.fixation import extract_fixations_from_df
    Sfix, Efix = extract_fixations_from_df(df)

Returns
-------
Sfix : list of [starttime]
Efix : list of [starttime, endtime, duration, x, y]
"""

import numpy as np
import pandas as pd

from igaze.detectors import fixation_detection


# Default column names matching config.yaml eyetracker_channels
_DEFAULT_X_COL   = "left_gaze_x"
_DEFAULT_Y_COL   = "left_gaze_y"
_DEFAULT_TIME_COL = "timestamp"


def extract_fixations(
    x: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    missing: float = 0.0,
    maxdist: int = 25,
    mindur: int = 50,
):
    """Detect fixations from numpy arrays.

    Parameters
    ----------
    x, y   : np.ndarray  — gaze positions
    time   : np.ndarray  — timestamps (milliseconds)
    missing: float       — value used for missing data (default 0.0)
    maxdist: int         — max inter-sample distance in pixels (default 25)
    mindur : int         — min fixation duration in ms (default 50)

    Returns
    -------
    Sfix : list of [starttime]
    Efix : list of [starttime, endtime, duration, x, y]
    """
    return fixation_detection(
        np.array(x, dtype=float),
        np.array(y, dtype=float),
        np.array(time, dtype=float),
        missing=missing,
        maxdist=maxdist,
        mindur=mindur,
    )


def extract_fixations_from_df(
    df: pd.DataFrame,
    x_col: str = _DEFAULT_X_COL,
    y_col: str = _DEFAULT_Y_COL,
    time_col: str = _DEFAULT_TIME_COL,
    average_eyes: bool = True,
    missing: float = 0.0,
    maxdist: int = 25,
    mindur: int = 50,
):
    """Detect fixations from an eyetracker DataFrame saved by xdf.py.

    Parameters
    ----------
    df           : pd.DataFrame  — eyetracker data (from parquet/csv)
    x_col        : str           — gaze x column (default 'left_gaze_x')
    y_col        : str           — gaze y column (default 'left_gaze_y')
    time_col     : str           — timestamp column (default 'timestamp')
    average_eyes : bool          — if True, average left+right gaze when both
                                   exist (right_gaze_x / right_gaze_y)
    missing      : float         — missing data value (default 0.0)
    maxdist      : int           — max inter-sample distance in pixels
    mindur       : int           — min fixation duration in ms

    Returns
    -------
    Sfix : list of [starttime]
    Efix : list of [starttime, endtime, duration, x, y]
    """
    if average_eyes and "right_gaze_x" in df.columns and "right_gaze_y" in df.columns:
        x = ((df[x_col] + df["right_gaze_x"]) / 2).values
        y = ((df[y_col] + df["right_gaze_y"]) / 2).values
    else:
        x = df[x_col].values
        y = df[y_col].values

    # convert LSL timestamps (seconds) to milliseconds for mindur compatibility
    time = df[time_col].values
    if time.mean() > 1e6:  # LSL timestamps are in seconds since epoch (~1e9)
        time = (time - time[0]) * 1000.0  # relative ms from start

    return extract_fixations(x, y, time, missing=missing, maxdist=maxdist, mindur=mindur)


def fixation_summary(Efix: list) -> dict:
    """Compute summary statistics from a list of fixations.

    Parameters
    ----------
    Efix : list of [starttime, endtime, duration, x, y]

    Returns
    -------
    dict with keys: count, total_duration_ms, mean_duration_ms, fixation_rate_per_sec
    """
    if not Efix:
        return {"count": 0, "total_duration_ms": 0.0,
                "mean_duration_ms": 0.0, "fixation_rate_per_sec": 0.0}

    durations = np.array([e[2] for e in Efix])
    total_ms  = durations.sum()

    return {
        "count":                  len(Efix),
        "total_duration_ms":      float(total_ms),
        "mean_duration_ms":       float(durations.mean()),
        "fixation_rate_per_sec":  len(Efix) / (total_ms / 1000.0) if total_ms > 0 else 0.0,
    }
