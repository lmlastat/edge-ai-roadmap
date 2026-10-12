import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/stress.csv"
out_path = sys.argv[2] if len(sys.argv) > 2 else "results/stress_plot.png"

df = pd.read_csv(csv_path)
df = df.iloc[1:]  # drop the first row: its CPU reading is a priming artifact

panels = [
    ("gpu_util_percent", "GPU utilization (%)"),
    ("power_w", "GPU power (W)"),
    ("temp_c", "GPU temperature (°C)"),
    ("vram_used_MB", "VRAM used (MB)"),
]

fig, axes = plt.subplots(len(panels), 1, figsize=(9, 8), sharex=True)

for ax, (col, label) in zip(axes, panels):
    ax.plot(df["t"], df[col], linewidth=1.5, color="#2a6f97")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

axes[-1].set_xlabel("time (s)")
fig.suptitle("sysprobe: 2-minute GPU stress run")
fig.tight_layout()
fig.savefig(out_path, dpi=150)
print("saved", out_path)