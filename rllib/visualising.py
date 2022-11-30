from ai_economist.foundation.utils import load_episode_log
import plotting  # plotting utilities for visualizing env. state
import matplotlib.pyplot as plt 

dense_log = load_episode_log("./tutorials/rllib/phase1/dense_logs/logs_0000000000640000/env003.lz4")

figs = plotting.breakdown(dense_log)

plt.show()