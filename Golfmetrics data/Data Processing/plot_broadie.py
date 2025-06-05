
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df_yards = pd.read_csv("/Users/federicadomecq/Documents/golfModeL47/broadiedata/strokes_by_lie_yards.csv")
df_green = pd.read_csv("/Users/federicadomecq/Documents/golfModeL47/broadiedata/strokes_on_green_feet.csv")

# Plotting strokes vs. distance (yards)
plt.figure(figsize=(10, 6))
for col in df_yards.columns[1:]:
    plt.plot(df_yards["Distance (yards)"], df_yards[col], label=col, lw=2)

plt.title("Average Strokes to Hole Out by Lie (Yards)")
plt.xlabel("Distance to Hole (yards)")
plt.ylabel("Average Strokes to Hole Out")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plot_strokes_by_lie_yards.png")
plt.close()

# Plotting strokes on the green (feet)
plt.figure(figsize=(8, 5))
plt.plot(df_green["Distance (feet)"], df_green["Green"], color='green', lw=2)
plt.title("Average Strokes to Hole Out on Green (Feet)")
plt.xlabel("Distance to Hole (feet)")
plt.ylabel("Average Strokes to Hole Out")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_strokes_on_green_feet.png")
plt.close()
