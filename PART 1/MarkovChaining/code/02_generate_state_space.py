import os
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely import wkt
import matplotlib.pyplot as plt

# === Load hole geometry ===
df = pd.read_csv("MarkovChaining/try1data/hole_1_data.csv")
df["geometry"] = df["WKT"].apply(wkt.loads)

# Debug: check bounds
for i, geom in enumerate(df["geometry"].head(5)):
    print(f"Geometry {i} bounds: {geom.bounds}")

# === Compute raster bounds ===
all_bounds = [geom.bounds for geom in df["geometry"]]
minx = min(b[0] for b in all_bounds)
miny = min(b[1] for b in all_bounds)
maxx = max(b[2] for b in all_bounds)
maxy = max(b[3] for b in all_bounds)

print("→ Geometry bounds:")
print(f"   x: [{minx:.2f}, {maxx:.2f}], y: [{miny:.2f}, {maxy:.2f}]")

# === Generate raster grid at 0.5-yard resolution ===
x_range = np.arange(minx - 1, maxx + 1, 3.0)
y_range = np.arange(miny - 1, maxy + 1, 3.0)
states = []

# === Assign lie type to each point ===
for x in x_range:
    for y in y_range:
        pt = Point(x, y)
        assigned = False
        for _, row in df.iterrows():
            if row["geometry"].contains(pt):
                states.append({"x": x, "y": y, "lie": row["lie"]})
                assigned = True
                break
        if not assigned:
            states.append({"x": x, "y": y, "lie": "rough"})

# === Save to CSV ===
states_df = pd.DataFrame(states)
os.makedirs("MarkovChaining/results/try1", exist_ok=True)
states_df.to_csv("MarkovChaining/results/try1/states.csv", index=False)
print(f"✅ Generated {len(states)} states. Saved to states.csv.")

# === Plot classified raster grid ===
plt.figure(figsize=(8, 8))
color_map = {
    "rough": "mediumseagreen",
    "fairway": "forestgreen",
    "green": "lightgreen",
    "bunker": "tan",
    "OB": "lightcoral",
    "tee": "darkgreen",
    "water_hazard": "skyblue"
}
for lie, group in states_df.groupby("lie"):
    plt.scatter(group["x"], group["y"], label=lie, color=color_map.get(lie, "gray"), s=2)

plt.gca().set_aspect("equal")
plt.legend()
plt.title("Rasterised Grid by Lie Type")
plt.tight_layout()
plt.show()