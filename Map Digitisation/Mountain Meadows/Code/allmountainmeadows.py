import pandas as pd
from shapely import wkt
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

# Load your exported CSV
df = pd.read_csv("/Users/federicadomecq/Desktop/Golf ModeL/Map Digitisation/Mountain Meadows/dataMM/golf_holes_full.csv")

# Convert geometry column
df["geometry"] = df["WKT"].apply(wkt.loads)

# Define your legend (color mapping)
lie_colors = {
    "bunker": "tan",
    "fairway": "forestgreen",
    "green": "lightgreen",
    "OB": "lightcoral",
    "rough": "mediumseagreen",
    "tee": "darkgreen",
    "water_hazard": "skyblue"
}

# Get all hole numbers excluding 19 (used for OB/rough)
holes = df["hole_ref"].dropna().unique()
holes = sorted([int(h) for h in holes if int(h) != 19])

for hole in holes:
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Get all main features for this hole
    hole_features = df[df["hole_ref"] == hole]
    
    # Get OB/rough (hole_ref == 19) that touch this hole
    background = df[(df["hole_ref"] == 19) & (df["lie"] != "rough")]
    main_union = unary_union(hole_features["geometry"].tolist())
    relevant_background = background[background["geometry"].apply(lambda g: g.intersects(main_union))]

    # Combine both
    full = pd.concat([hole_features, relevant_background])

    # Plot by lie
    for _, row in full.iterrows():
        geom = row["geometry"]
        color = lie_colors.get(row["lie"], "gray")
        parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        for poly in parts:
            x, y = poly.exterior.xy
            ax.fill(x, y, color=color, label=row["lie"], alpha=0.75)

    # Clean legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Add legend with white background box and no overlap
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=1,
        edgecolor="black"
    )

    # Title and style
    ax.set_title(f"Hole {hole} Layout")
    ax.set_aspect("equal")

    # Show lat/lon axes
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    # Final formatting
    plt.tight_layout()
    plt.savefig(f"/Users/federicadomecq/Desktop/Golf ModeL/Map Digitisation/Mountain Meadows Layouts/True Orientation/hole_{hole}_layout.png", dpi=300)
    plt.close()
