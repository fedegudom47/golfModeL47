import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.affinity import rotate

'''
Getting hole layouts from OSM data in WKT format, translating to aim upwards
'''


# === Load Data ===
df = pd.read_csv("/Users/federicadomecq/Desktop/Golf ModeL/Map Digitisation/Mountain Meadows/dataMM/golf_holes_full.csv")
df["geometry"] = df["WKT"].apply(wkt.loads)

lines = pd.read_csv("hole_lines.csv")
lines["geometry"] = lines["WKT"].apply(wkt.loads)

# === Color mapping ===
lie_colors = {
    "bunker": "tan",
    "fairway": "forestgreen",
    "green": "lightgreen",
    "OB": "lightcoral",
    "rough": "mediumseagreen",
    "tee": "darkgreen",
    "water_hazard": "skyblue"
}

# === Get hole numbers ===
holes = df["hole_ref"].dropna().unique()
holes = sorted([int(h) for h in holes if int(h) != 19])

# === Store rotated results ===
rotated_rows = []

# === Loop through each hole ===
for hole in holes:
    fig, ax = plt.subplots(figsize=(6, 6))

    # Get features and background
    hole_features = df[df["hole_ref"] == hole]
    background = df[(df["hole_ref"] == 19) & (df["lie"] != "rough")]
    main_union = unary_union(hole_features["geometry"].tolist())
    relevant_background = background[background["geometry"].apply(lambda g: g.intersects(main_union))]

    # Get line geometry for hole direction
    line_geom = lines[lines["ref"] == hole]["geometry"].values
    if len(line_geom) == 0:
        print(f"Skipping hole {hole}: no line found")
        continue

    line_geom = line_geom[0]
    p1, p2 = line_geom.coords[0], line_geom.coords[1]
    vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])

    # Calculate angle to face up (north = 90 degrees)
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    rotate_by = 90 - angle

    # Rotate geometries
    combined = pd.concat([hole_features, relevant_background]).copy()
    combined["geometry"] = combined["geometry"].apply(lambda g: rotate(g, rotate_by, origin=p1))

    # Save to plot
    for _, row in combined.iterrows():
        color = lie_colors.get(row["lie"], "gray")
        geom = row["geometry"]
        parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        for poly in parts:
            x, y = poly.exterior.xy
            ax.fill(x, y, color=color, label=row["lie"], alpha=0.75)

    # Save rotated features to master list
    rotated_rows.append(combined)

    # Format plot
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=True, facecolor="white", edgecolor="black", framealpha=1
    )

    ax.set_title(f"Hole {hole} Layout (Facing Up)")
    ax.set_aspect("equal")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"/Users/federicadomecq/Desktop/Golf ModeL/Map Digitisation/Mountain Meadows Layouts/Upwards/hole_{hole}_rotated_upward.png", dpi=300)
    plt.close()

# === Export all rotated geometries ===
final_df = pd.concat(rotated_rows)
final_df["WKT"] = final_df["geometry"].apply(lambda g: g.wkt)
final_df.drop(columns=["geometry"], inplace=True)
final_df.to_csv("/Users/federicadomecq/Desktop/rotated_golf_holes.csv", index=False)
print("✅ Saved rotated_golf_holes.csv")
