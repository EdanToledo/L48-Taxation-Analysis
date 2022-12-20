from ai_economist.foundation.utils import load_episode_log
from utils import plotting  # plotting utilities for visualizing env. state
import matplotlib.pyplot as plt 

free_market_folder = "free-market-dummy-results"
taxes_folder = "us-taxes-dummy-results"

taxes_dense_log = load_episode_log(f"./rllib_code/{taxes_folder}/dense_logs/logs_0000000024600000/env003.lz4")
free_market_dense_log = load_episode_log(f"./rllib_code/{free_market_folder}/dense_logs/logs_0000000024600000/env003.lz4")


# figs = plotting.breakdown(taxes_dense_log)
# figs = plotting.breakdown(free_market_dense_log)

plt.show()