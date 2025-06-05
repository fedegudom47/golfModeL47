import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

lies = ['fairway', 'sand', 'deep_rough', 'rough']  # will handle tee separately

os.makedirs("results", exist_ok=True)

for lie in lies:
    print(f"Processing {lie}...")

    df = pd.read_csv(f"/Users/federicadomecq/Documents/golfModeL47/Golfmetrics data/cleaned_shots/all_lies_data/shots_from_{lie}.csv")

    # Drop rows with missing data just in case
    df = df.dropna(subset=['holedis', 'shots_to_hole_out'])

    # Filter distances under 250 
    df = df[df['holedis'] < 250]

    # Bin distances (e.g., every 5 yards)
    df['bin'] = (df['holedis'] // 5) * 5
    grouped = df.groupby('bin').agg(
        avg_strokes=('shots_to_hole_out', 'mean'),
        count=('shots_to_hole_out', 'count')
    ).reset_index().rename(columns={'bin': 'holedis'})

    # Remove bins with very few samples 
    grouped = grouped[grouped['count'] >= 3]

    X = grouped[['holedis']].values
    y = grouped['avg_strokes'].values

    kernel = RBF(length_scale_bounds=(5, 100.0)) + WhiteKernel(noise_level=0.05, noise_level_bounds=(0.05, 1.0))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=True)
    gpr.fit(X, y)

    print(f"Optimised kernel for {lie}: {gpr.kernel_}")

    X_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_pred, std_pred = gpr.predict(X_grid, return_std=True)

    preds = pd.DataFrame({
        'holedis': X_grid.flatten(),
        'pred': y_pred,
        'std': std_pred
    })
    preds.to_csv(f"results/gpr_{lie}_preds.csv", index=False)

    # --------- Plot ---------
    plt.figure(figsize=(8, 5))
    plt.plot(preds["holedis"], preds["pred"], label="GPR prediction", lw=2)
    plt.fill_between(preds["holedis"],
                     preds["pred"] - preds["std"],
                     preds["pred"] + preds["std"],
                     color="lightblue", alpha=0.4, label="±1 std. dev")
    plt.scatter(grouped["holedis"], grouped["avg_strokes"], color="black", s=40, label="Binned avg data")
    plt.title(f"GPR Prediction - {lie.capitalize()}")
    plt.xlabel("Distance to hole (yards)")
    plt.ylabel("Predicted strokes to hole out")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/gpr_{lie}_plot.png")
    plt.close()

    print(f"Saved predictions and plot for {lie}.\n")
