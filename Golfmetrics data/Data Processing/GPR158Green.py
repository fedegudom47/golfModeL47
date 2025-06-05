import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Load raw putting data
df = pd.read_csv("/Users/federicadomecq/Documents/golfModeL47/Golfmetrics data/cleaned_shots/green_data_feet/shots_from_green_feet.csv")

# Drop missing values
df = df.dropna(subset=['holedis', 'shots_to_hole_out'])

# Filter to max 100 feet
df = df[df['holedis'] <= 90]

# Bin by every 1 foot
df['bin'] = (df['holedis'] // 1).astype(int)

# Aggregate
grouped = df.groupby('bin').agg(
    avg_strokes=('shots_to_hole_out', 'mean'),
    count=('shots_to_hole_out', 'count')
).reset_index().rename(columns={'bin': 'feet'})

# Filter out sparse bins
grouped = grouped[grouped['count'] >= 10]

# Train GPR
X = grouped[['feet']].values
y = grouped['avg_strokes'].values

kernel = RBF(length_scale_bounds=(1.0, 20.0)) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-4, 0.5))
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True)
gpr.fit(X, y)

print(f"Optimised kernel for putting: {gpr.kernel_}")

# Predict on grid
X_grid = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_pred, std_pred = gpr.predict(X_grid, return_std=True)

# Save predictions
preds = pd.DataFrame({
    'feet': X_grid.flatten(),
    'pred': y_pred,
    'std': std_pred
})
os.makedirs("/Users/federicadomecq/Documents/golfModeL47/Golfmetrics data/results158code", exist_ok=True)
preds.to_csv("/Users/federicadomecq/Documents/golfModeL47/Golfmetrics data/results158code/gpr_green_from_raw_preds.csv", index=False)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(preds["feet"], preds["pred"], label="GPR prediction", lw=2, color="green")
plt.fill_between(preds["feet"], preds["pred"] - preds["std"], preds["pred"] + preds["std"],
                 color="lightgreen", alpha=0.4, label="±1 std. dev")
plt.scatter(grouped["feet"], grouped["avg_strokes"], color="black", s=40, label="Binned avg data")
plt.title("GPR Prediction – Putting (Feet)")
plt.xlabel("Distance to hole (feet)")
plt.ylabel("Predicted strokes to hole out")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("/Users/federicadomecq/Documents/golfModeL47/Golfmetrics data/results158code/gpr_green_from_raw_plot.png")
plt.close()

print("Saved predictions and plot for putting data.")
