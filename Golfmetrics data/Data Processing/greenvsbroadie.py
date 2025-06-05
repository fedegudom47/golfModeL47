import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Load GPR predictions (your stats) ---
gpr_df = pd.read_csv("/Users/federicadomecq/Documents/golfModeL47/Golfmetrics data/results158code/gpr_green_from_raw_preds.csv")  # update path if needed

# --- Load benchmark stats (tour average) ---
benchmark_df = pd.read_csv("/Users/federicadomecq/Documents/golfModeL47/broadiedata/strokes_on_green_feet_broadie.csv")
benchmark_df.columns = ["feet", "benchmark"]

# --- Plotting ---
plt.figure(figsize=(9, 5))
plt.plot(benchmark_df["feet"], benchmark_df["benchmark"], label="Tour Benchmark", color="gray", linestyle="--")
plt.plot(gpr_df["feet"], gpr_df["pred"], label="GPR Prediction", color="green", lw=2)
plt.fill_between(gpr_df["feet"], gpr_df["pred"] - gpr_df["std"], gpr_df["pred"] + gpr_df["std"],
                 color="lightgreen", alpha=0.4, label="±1 std. dev")
plt.title("Putting Performance vs Tour Benchmark")
plt.xlabel("Distance to Hole (feet)")
plt.ylabel("Average Strokes to Hole Out")
plt.grid(True)
plt.legend()
plt.tight_layout()

# --- Save output ---
plt.savefig("/Users/federicadomecq/Documents/golfModeL47/Golfmetrics data/results158code/compare_putting_to_benchmark.png")
plt.close()

print("✅ Plot saved as:/results158code/compare_putting_to_benchmark.png")
