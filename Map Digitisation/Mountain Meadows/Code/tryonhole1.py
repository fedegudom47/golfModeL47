import pandas as pd
from shapely import wkt
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon

'''
First attempt at going from qgis to shapely
'''

# Load your exported CSV
df = pd.read_csv("/Users/federicadomecq/Desktop/att1.csv")

# Convert WKT string to Shapely geometry
df["geometry"] = df["WKT"].apply(wkt.loads)

# Filter for Hole 1 only
hole1 = df[df["hole_n"] == 1]

# Plot the features
fig, ax = plt.subplots(figsize=(8, 8))

# Optional: use different colours for each 'real_lie' type
colors = {"fw": "green", "fairway": "green","bunker": "sandybrown", "teebox" :"darkgreen", "green" :"palegreen", "h2o":"skyblue"}
for _, row in hole1.iterrows():
    geom = row["geometry"]
    lie = row["real_lie"]
    color = colors.get(lie, "gray")

    # Handle both Polygon and MultiPolygon
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for poly in polys:
        x, y = poly.exterior.xy
        ax.fill(x, y, color=color, alpha=0.6, label=lie)

# Clean up legend and axes
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # remove duplicates
ax.legend(by_label.values(), by_label.keys())
ax.set_title("Hole 1 Layout")
ax.set_aspect("equal")
plt.show()
