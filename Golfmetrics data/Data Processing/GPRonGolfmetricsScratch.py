import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------------------------------
# 🧮 RBF Kernel (Radial Basis Function)
# ----------------------------------------
# Measures similarity between two input points (distances).
# The closer x and y are, the higher the value (max = 1 if identical).
# This controls the "smoothness" of the GP function.
def rbf_kernel(x, y, sigma):
    return np.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2))


# ----------------------------------------
# 🤖 Gaussian Process Regression Function
# ----------------------------------------
# Given training data (x, y), noise level, and kernel bandwidth,
# it computes the posterior predictive distribution over new points.
def gpreg(x, y, lam, sig, design):
    n = len(x)  # Number of training observations
    m = len(design)  # Number of test points (grid over distance)

    # Combine training and prediction points into one big array
    all_points = np.concatenate((x, design))
    N = n + m  # Total number of points

    # Build the full kernel (covariance) matrix over all points
    Sigma = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            k_val = rbf_kernel(np.array([all_points[i]]), np.array([all_points[j]]), lam)
            Sigma[i, j] = k_val
            Sigma[j, i] = k_val  # Symmetric matrix

    # Partition the kernel matrix:
    # S11: cov(train, train)
    # S12: cov(train, test)
    # S21: cov(test, train)
    # S22: cov(test, test)
    S11 = Sigma[:n, :n]
    S12 = Sigma[:n, n:]
    S21 = Sigma[n:, :n]
    S22 = Sigma[n:, n:]

    # Add noise variance (sig² * I) to training data covariance
    noise_matrix = sig**2 * np.eye(n)

    # Compute posterior mean: μ* = S21 · (S11 + σ²I)⁻¹ · y
    inv = S21 @ np.linalg.inv(S11 + noise_matrix)
    mean = inv @ y

    # Compute posterior covariance: Σ* = S22 - S21 · (S11 + σ²I)⁻¹ · S12
    cov = S22 - inv @ S12
    vars = np.diag(cov)  # Only extract variances

    return mean, vars


# ----------------------------------------
# 🗂️ File paths and settings
# ----------------------------------------

# Preprocessed lie-specific shot files (output from your cleaning script)
lie_files = {
    "tee": "/Users/federicadomecq/Documents/golfModeL47/Golfmetrics data/cleaned_shots/all_lies_data/shots_from_tee.csv",
    "fairway": "/Users/federicadomecq/Documents/golfModeL47/Golfmetrics data/cleaned_shots/all_lies_data/shots_from_fairway.csv",
    "rough": "/Users/federicadomecq/Documents/golfModeL47/Golfmetrics data/cleaned_shots/all_lies_data/shots_from_rough.csv",
    "sand": "/Users/federicadomecq/Documents/golfModeL47/Golfmetrics data/cleaned_shots/all_lies_data/shots_from_sand.csv",
    "deep_rough": "/Users/federicadomecq/Documents/golfModeL47/Golfmetrics data/cleaned_shots/all_lies_data/shots_from_deep_rough.csv"
}

# 🔧 Hyperparameters for the GP kernel and noise model
length_scale = 0.3     # Controls how "wiggly" the function is — smaller = more wiggly
noise_sigma = 0.2      # Assumed noise in shot-to-hole-out values

bin_size = 5         # Yard-wide binning for aggregation

# Output folder for your plots
output_dir = "/Users/federicadomecq/Documents/golfModeL47/Golfmetrics data/gpr_plots_binned"
os.makedirs(output_dir, exist_ok=True)


# ----------------------------------------
# 🔁 Loop over each lie and run GPR
# ----------------------------------------

for lie, file_path in lie_files.items():
    # Load CSV for this lie type
    df = pd.read_csv(file_path)

    # FILTER: Limit distance range depending on lie
    if lie == "tee":
        df = df[df["holedis"] >= 80]  # For tee shots, keep only ≤ 80 yards
    else:
        df = df[df["holedis"] <= 250]  # For all others, keep only ≤ 250 yards

    # Skip if nothing left after filtering
    if df.empty:
        print(f"⚠️ Skipping {lie} — no data in desired distance range.")
        continue


    # 🧺 BINNING: round holedis to the nearest lower multiple of `bin_size`
    df["dist_bin"] = (df["holedis"] // bin_size) * bin_size

    # 📊 GROUP BY BIN: compute mean strokes to hole out and observation count
    grouped = df.groupby("dist_bin").agg(
        y_mean=("shots_to_hole_out", "mean"),   # Average strokes in that bin
        n_obs=("shots_to_hole_out", "count")    # How many shots went into it
    ).reset_index()

    # 🚫 FILTER: remove sparse bins (e.g., < 5 shots)
    grouped = grouped[grouped["n_obs"] >= 5]

    # 🎯 Set up x and y for GPR (scaled to ~0–1 for stability)
    x = grouped["dist_bin"].values / 300          # Scaled distance
    y = grouped["y_mean"].values                  # Mean strokes
    counts = grouped["n_obs"].values              # For plotting marker size

    # 🎨 Design grid: evenly spaced test points for prediction
    design = np.linspace(min(x), max(x), 200)

    # 🤖 Run GPR
    mean, vars = gpreg(x, y, lam=length_scale, sig=noise_sigma, design=design)

    # ----------------------------------------
    # 📈 Plotting the GPR output
    # ----------------------------------------

    plt.figure(figsize=(10, 6))

    # GP Mean prediction curve
    plt.plot(design * 300, mean, label='GPR Mean', linewidth=2)

    # Confidence interval (±2 std dev)
    plt.fill_between(design * 300,
                     mean - 2 * np.sqrt(vars),
                     mean + 2 * np.sqrt(vars),
                     color='lightblue', alpha=0.3,
                     label='±2 SD')

    # Plot binned data: size = how many shots in each bin
    plt.scatter(grouped["dist_bin"], y,
                s=counts, alpha=0.8, color='black', label='Binned Averages')

    # 🧾 Annotate each point with number of shots used (optional)
    for i, row in grouped.iterrows():
        plt.text(row["dist_bin"], row["y_mean"] + 0.05,
                 f"{int(row['n_obs'])}", fontsize=8, ha='center', alpha=0.6)

    # Labels and layout
    plt.title(f"GPR (Binned): Shots to Hole Out vs Distance — {lie.replace('_', ' ').title()}")
    plt.xlabel("Distance to Hole (yards)")
    plt.ylabel("Shots to Hole Out")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 💾 Save plot
    filename = os.path.join(output_dir, f"gpr_{lie}_binned.png")
    plt.savefig(filename)
    plt.close()

    # ✅ Confirm output
    print(f"✅ Saved plot for {lie} — using {len(grouped)} distance bins")
