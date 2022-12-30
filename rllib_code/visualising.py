from ai_economist.foundation.utils import load_episode_log
from utils import plotting  # plotting utilities for visualizing env. state
import matplotlib.pyplot as plt 

folder_name = "free-market-50-spacing"
taxes_dense_log = load_episode_log(f"/Users/edantoledo/Documents/MLandthePhysicalWorld/FinalProject/{folder_name}/dense_logs/logs_0000000048000000/env002.lz4")

figs = plotting.breakdown(taxes_dense_log)

plt.show()