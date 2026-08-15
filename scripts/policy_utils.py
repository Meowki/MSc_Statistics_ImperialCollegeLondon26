import numpy as np
import pandas as pd


TIE_HASH_KEY = "0123456789123456"
LABELLED_DAYS = (0, 29)
LABELLED_DAYS_COUNT = 18


def add_tie_key(data):
    data = data.copy()
    data["_tie_key"] = pd.util.hash_pandas_object(
        data[["src_user", "window_id"]],
        index=False,
        hash_key=TIE_HASH_KEY,
    ).to_numpy(dtype=np.uint64)
    return data


def labelled_metrics(alerts, account_type):
    labelled_alerts = int(
        alerts["day"].between(LABELLED_DAYS[0], LABELLED_DAYS[1]).sum()
    )
    if account_type != "human":
        return {
            "labelled_days_hit": np.nan,
            "labelled_days_total": np.nan,
            "labelled_day_overlap": np.nan,
            "labelled_windows_hit": np.nan,
            "total_alerts_labelled": labelled_alerts,
        }

    hits = alerts[alerts["redteam_user_window_flag"] == 1]
    days_hit = int(hits["day"].nunique())
    return {
        "labelled_days_hit": days_hit,
        "labelled_days_total": LABELLED_DAYS_COUNT,
        "labelled_day_overlap": days_hit / LABELLED_DAYS_COUNT,
        "labelled_windows_hit": len(hits),
        "total_alerts_labelled": labelled_alerts,
    }
