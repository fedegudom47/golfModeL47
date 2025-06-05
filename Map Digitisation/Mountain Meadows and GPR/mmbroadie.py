import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import Point, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.affinity import rotate
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# === Load data ===
df = pd.read_csv("Map Digitisation/Mountain Meadows/dataMM/golf_holes_full.csv")
df["geometry"] = df["WKT"].apply(wkt.loads)

df["lie"] = df["lie"].str.strip().str.lower()


lines = pd.read_csv("Map Digitisation/Mountain Meadows/dataMM/hole_lines.csv")
lines["geometry"] = lines["WKT"].apply(wkt.loads)


yards = pd.read_csv("broadiedata/strokes_by_lie_yards_broadie.csv")
feet = pd.read_csv("broadiedata/strokes_on_green_feet_broadie.csv")
feet["Distance (yards)"] = feet["Distance (feet)"] / 3
feet = feet.drop(columns=["Distance (feet)"])
green_interp = pd.Series(feet["Green"].values, index=feet["Distance (yards)"].values)

yard_interp = {
    col.lower(): pd.Series(yards[col].values, index=yards["Distance (yards)"])
    for col in ["Tee", "Fairway", "Rough", "Sand", "Recovery"]
}
yard_interp["green"] = green_interp

# === Set hole ===
hole = 1
hole_df = df[df["hole_ref"] == hole].copy()
line_geom = lines[lines["ref"] == hole]["geometry"].values[0]
p1, p2 = line_geom.coords[0], line_geom.coords[1]
angle = 90 - np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
hole_df["geometry"] = hole_df["geometry"].apply(lambda g: rotate(g, angle, origin=p1))

green_union = unary_union(hole_df[hole_df["lie"] == "green"]["geometry"].tolist())
pin = green_union.centroid

# === Radial grid (centered at pin) ===
r_vals = np.linspace(0, 150, 300)
theta_vals = np.linspace(0, 2 * np.pi, 360)
R, T = np.meshgrid(r_vals, theta_vals)
X = R * np.cos(T)
Y = R * np.sin(T)

# === Utility functions ===
def lookup_lie(x, y):
    point = Point(x + pin.x, y + pin.y)

    # Map any extra labels to Broadie-compatible ones
    lie_aliases = {
        "bunker": "sand",
        "green": "green",
        "fairway": "fairway",
        "rough": "rough",
        "tee": "tee"
    }

    for raw_lie, broadie_lie in lie_aliases.items():
        match = hole_df[hole_df["lie"] == raw_lie]
        if any(g.contains(point) for g in match["geometry"]):
            return broadie_lie
    return None


def get_strokes(r, lie):
    if lie not in yard_interp:
        return np.nan
    series = yard_interp[lie].interpolate(method="linear", limit_direction="both")
    return series.reindex([r], method="nearest").iloc[0]

def is_inside_course(x, y):
    pt = Point(x + pin.x, y + pin.y)
    return any(g.contains(pt) for g in hole_df["geometry"])

# === Compute Z values (masked to course features) ===
Z = np.full_like(R, np.nan)
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        x, y = X[i, j], Y[i, j]
        if not is_inside_course(x, y):
            continue
        r = R[i, j]
        lie = lookup_lie(x, y)
        if lie:
            Z[i, j] = get_strokes(r, lie)

# === Plot ===
fig, ax = plt.subplots(figsize=(8, 8))
norm = Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z))
mesh = ax.pcolormesh(X, Y, Z, shading='auto', cmap='inferno_r', norm=norm)

# Draw dashed rings
for radius in range(10, 160, 10):
    ax.add_patch(plt.Circle((0, 0), radius, color='black', fill=False, lw=0.5, ls='--', alpha=0.4))

# Draw course outlines
for _, row in hole_df.iterrows():
    geom = row["geometry"]
    parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for poly in parts:
        x = np.array(poly.exterior.xy[0]) - pin.x
        y = np.array(poly.exterior.xy[1]) - pin.y
        ax.plot(x, y, color='black', linewidth=0.7)

# Draw pin marker
ax.text(0, 0, 'X', fontsize=16, ha='center', va='center', color='red', weight='bold')

# Final formatting
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(f"Hole {hole} — Radial Strokes Heatmap", fontsize=14)

sm = ScalarMappable(cmap='inferno_r', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.04)
cbar.set_label("Strokes to Hole Out")

plt.tight_layout()
plt.savefig(f"Map Digitisation/Mountain Meadows and GPR/hole_{hole}_masked_radial.png", dpi=300)
plt.show()
