from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from plotstyle import *


WINDOWS_FILE = Path("outputs/tables/user_window_counts.parquet")
REDTEAM_FILE = Path("outputs/tables/redteam_user_window_labels.parquet")
WINDOWS_PER_DAY = 288


windows = pd.read_parquet(
    WINDOWS_FILE,
    columns=["window_id", "event_count"],
)
windows["day"] = windows["window_id"] // WINDOWS_PER_DAY
daily_volume = windows.groupby("day")["event_count"].sum() / 1e6

redteam = pd.read_parquet(REDTEAM_FILE, columns=["day"])
daily_redteam = redteam.groupby("day").size()

fig, ax = plt.subplots(figsize=(FULL_W, SINGLE_H))

# red-team counts
label_axis = ax.twinx()
label_axis.bar(
    daily_redteam.index,
    daily_redteam.values,
    width=0.8,
    color=TANGERINE,
    alpha=0.35,
)
label_axis.set_ylabel("Red-team user-windows")
label_axis.set_ylim(0, 300)
label_axis.spines["top"].set_visible(False)
label_axis.spines["right"].set_visible(True)

# authentication volume
line, = ax.plot(
    daily_volume.index,
    daily_volume.values,
    color=NAVY,
    label="Daily authentication volume",
)
ax.set_xlabel("Day")
ax.set_ylabel("Authentication events (millions)")
ax.set_xlim(daily_volume.index.min() - 0.5, daily_volume.index.max() + 0.5)
ax.set_zorder(label_axis.get_zorder() + 1)
ax.patch.set_visible(False)

bar = Patch(facecolor=TANGERINE, alpha=0.35, label="Daily red-team count")
ax.legend(handles=[line, bar], loc="upper left")

save_fig(fig, "data_overview")
plt.close(fig)
print("saved data_overview")
