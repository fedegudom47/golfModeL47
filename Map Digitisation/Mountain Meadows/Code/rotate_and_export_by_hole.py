import pandas as pd
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.geometry import MultiPolygon
import os

# === SETTINGS ===
INPUT_CSV = "Map Digitisation/Mountain Meadows/dataMM/golf_holes_yardage.csv"
OUTPUT_DIR = "Map Digitisation/Mountain Meadows/MountainMeadows_Separated"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(INPUT_CSV)
df["geometry"] = df["WKT"].apply(wkt.loads)

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

# === Plot and save each hole ===
holes = sorted([int(h) for h in df["hole_ref"].dropna().unique() if int(h) != 19])

for hole in holes:
    hole_dir = f"{OUTPUT_DIR}/hole_{hole}"
    os.makedirs(hole_dir, exist_ok=True)

    hole_df = df[df["hole_ref"] == hole]

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    for _, row in hole_df.iterrows():
        geom = row["geometry"]
        lie = row["lie"]
        color = lie_colors.get(lie, "gray")

        parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        for poly in parts:
            x, y = poly.exterior.xy
            ax.fill(x, y, color=color, label=lie, alpha=0.75)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        frameon=True, facecolor="white", edgecolor="black", framealpha=1
    )

    ax.set_title(f"Hole {hole} Layout (Yardage Aligned)", fontsize=14)
    ax.set_xlabel("Yards (Horizontal)")
    ax.set_ylabel("Yards (Up the Hole)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"{hole_dir}/hole_{hole}_layout.png", dpi=300)
    plt.close()

    # Save CSV
    hole_df["WKT"] = hole_df["geometry"].apply(lambda g: g.wkt)
    hole_df.drop(columns=["geometry"], inplace=True)
    hole_df.to_csv(f"{hole_dir}/hole_{hole}_data.csv", index=False)

print(f"✅ All hole layouts and CSVs saved to: {OUTPUT_DIR}")
