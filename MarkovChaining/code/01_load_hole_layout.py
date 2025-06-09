import pandas as pd
from shapely import wkt
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon

# Load data
df = pd.read_csv("MarkovChaining/try1data/hole_1_data.csv")
df["geometry"] = df["WKT"].apply(wkt.loads)

def plot_layout(df):
    fig, ax = plt.subplots(figsize=(8, 8))

    for _, row in df.iterrows():
        g = row["geometry"]
        lie = row.get("lie") or row.get("lie_type") or "unknown"

        # Plot single polygon
        if isinstance(g, Polygon):
            x, y = g.exterior.xy
            ax.fill(x, y, alpha=0.5, label=lie)

        # Plot multipolygon
        elif isinstance(g, MultiPolygon):
            for poly in g.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.5, label=lie)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")

    ax.set_title("Hole 1 Layout")
    ax.set_aspect("equal")
    ax.grid(True)
    plt.show()

plot_layout(df)
