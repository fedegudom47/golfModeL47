import pandas as pd
import os

# Load your CSV file
file_path = "/Users/federicadomecq/Documents/golfModeL47/Golfmetrics data/ppdatacomplete.csv"
df = pd.read_csv(file_path)

# Create output directories
base_dir = "cleaned_shots"
folders = {
    "all_lies_data": os.path.join(base_dir, "all_lies_data"),
    "green_data_yards": os.path.join(base_dir, "green_data_yards"),
    "green_data_feet": os.path.join(base_dir, "green_data_feet")
}
for path in folders.values():
    os.makedirs(path, exist_ok=True)

# Drop rows with missing essential data
essential_cols = ['roundid', 'holeid', 'shotid', 'stroke', 'startpos', 'holedis']
df_clean = df.dropna(subset=essential_cols)

# Convert startpos and stroke to int
df_clean['startpos'] = df_clean['startpos'].astype(int)
df_clean['stroke'] = df_clean['stroke'].astype(int)

# Filter out pickup shots and zero/negative distances
df_clean = df_clean[(df_clean['pickup'] != 1) & (df_clean['holedis'] > 0)]

# Compute shots-to-hole-out
df_clean["max_stroke_in_hole"] = df_clean.groupby(["roundid", "holeid"])["stroke"].transform("max")
df_clean["shots_to_hole_out"] = df_clean["max_stroke_in_hole"] - df_clean["stroke"] + 1

# 🔪 Keep only essential columns
df_clean = df_clean[[
    "roundid", "holeid", "hnum", "shotid", "stroke", "startpos", "holedis", "shots_to_hole_out"
]]

# Map start positions to descriptive names
lie_names = {
    0: "tee",
    1: "fairway",
    2: "rough",
    3: "sand",
    4: "green",
    6: "deep_rough"
}

# Save non-green shots by lie
all_lies_data = df_clean[df_clean['startpos'] != 4]
for code, name in lie_names.items():
    if code == 4:
        continue
    subset = all_lies_data[all_lies_data['startpos'] == code]
    subset.to_csv(f"{folders['all_lies_data']}/shots_from_{name}.csv", index=False)

# Handle green shots (in feet and yards)
green = df_clean[df_clean['startpos'] == 4].copy()
green['holedis_yards'] = green['holedis'] / 3.0

# Save green shots
green.to_csv(f"{folders['green_data_feet']}/shots_from_green_feet.csv", index=False)
green.to_csv(f"{folders['green_data_yards']}/shots_from_green_yards.csv", index=False)
