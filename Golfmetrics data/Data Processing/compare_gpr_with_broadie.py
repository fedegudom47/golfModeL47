
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load benchmark data
df_yards = pd.read_csv("/Users/federicadomecq/Documents/golfModeL47/broadiedata/strokes_by_lie_yards_broadie.csv")
lies = ["tee", "fairway", "rough", "sand", "recovery"]

# Overlay GPR predictions
for lie in lies:
    try:
        preds = pd.read_csv(f"/Users/federicadomecq/Documents/golfModeL47/Golfmetrics data/results158code/gpr_{lie}_preds.csv")
    except FileNotFoundError:
        print(f"No GPR file found for {lie}, skipping.")
        continue

    # Plot
    plt.figure(figsize=(10, 6))

    # Reference stroke curve
    plt.plot(df_yards["Distance (yards)"], df_yards[lie.capitalize()], label=f"{lie.capitalize()} (Benchmark)", linestyle="--", color="gray")

    # GPR prediction
    plt.plot(preds["holedis"], preds["pred"], label="GPR Prediction", lw=2)
    plt.fill_between(preds["holedis"], preds["pred"] - preds["std"], preds["pred"] + preds["std"], alpha=0.3, label="±1 std. dev")

    plt.title(f"GPR vs Benchmark: {lie.capitalize()}")
    plt.xlabel("Distance to Hole (yards)")
    plt.ylabel("Predicted Strokes to Hole Out")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"/Users/federicadomecq/Documents/golfModeL47/Golfmetrics data/results158code/gpr_comparison_{lie}.png")
    plt.close()
