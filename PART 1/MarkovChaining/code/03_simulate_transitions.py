# import pandas as pd
# import numpy as np
# from shapely.geometry import Point
# from math import radians, cos, sin

# # Load dispersion data
# disp_df = pd.read_csv("MarkovChaining/try1data/simulated_lpga_shot_data.csv")  # Needs club, dx, dy per shot
# states = pd.read_csv("MarkovChaining/results/try1/states.csv")

# # Simulate transitions for ALL clubs from tee
# start_x, start_y, start_lie = 0, 0, "tee"
# aim_deg = 0  # adjust globally if you want to aim left/right

# def match_state(x, y, states_df):
#     df = states_df.copy()
#     df["dist"] = np.sqrt((df["x"] - x)**2 + (df["y"] - y)**2)
#     return df.loc[df["dist"].idxmin(), ["x", "y", "lie"]].to_dict()

# all_transitions = []
# clubs = disp_df["club"].unique()

# for club in clubs:
#     shots = disp_df[disp_df["club"] == club].copy()
#     if shots.empty:
#         continue

#     # Rotate shot dispersions by aim
#     theta = radians(aim_deg)
#     shots["dx_rot"] = shots["dx"] * cos(theta) - shots["dy"] * sin(theta)
#     shots["dy_rot"] = shots["dx"] * sin(theta) + shots["dy"] * cos(theta)
#     shots["x_final"] = start_x + shots["dx_rot"]
#     shots["y_final"] = start_y + shots["dy_rot"]

#     for _, row in shots.iterrows():
#         result = match_state(row["x_final"], row["y_final"], states)
#         result["club"] = club
#         all_transitions.append(result)

# # Save all transitions
# pd.DataFrame(all_transitions).to_csv("MarkovChaining/results/try1/sample_transitions_all_clubs.csv", index=False)
# print(f"✅ Saved transitions for {len(clubs)} clubs.")
import pandas as pd
import numpy as np
from shapely.geometry import Point
from math import radians, cos, sin

# Load dispersion data and state grid
disp_df = pd.read_csv("MarkovChaining/try1data/simulated_lpga_shot_data.csv")
states = pd.read_csv("MarkovChaining/results/try1/states.csv")

# Starting state for all shots (tee box)
start_x, start_y, start_lie = 0, 0, "tee"
aim_deg = 0  # Adjust this to simulate aim left/right

def match_state(x, y, states_df):
    df = states_df.copy()
    df["dist"] = np.sqrt((df["x"] - x)**2 + (df["y"] - y)**2)
    return df.loc[df["dist"].idxmin(), ["x", "y", "lie"]].to_dict()

# Prepare transition list
all_transitions = []
clubs = disp_df["club"].unique()

for club in clubs:
    shots = disp_df[disp_df["club"] == club].copy()
    if shots.empty:
        continue

    # Rotate dispersion by aim angle
    theta = radians(aim_deg)
    shots["dx_rot"] = shots["dx"] * cos(theta) - shots["dy"] * sin(theta)
    shots["dy_rot"] = shots["dx"] * sin(theta) + shots["dy"] * cos(theta)
    shots["x_final"] = start_x + shots["dx_rot"]
    shots["y_final"] = start_y + shots["dy_rot"]

    for _, row in shots.iterrows():
        result = match_state(row["x_final"], row["y_final"], states)
        result["club"] = club
        result["x0"] = start_x
        result["y0"] = start_y
        result["lie0"] = start_lie
        all_transitions.append(result)

# Save to CSV
pd.DataFrame(all_transitions).to_csv("MarkovChaining/results/try1/sample_transitions_all_clubs.csv", index=False)
print(f"✅ Saved {len(all_transitions)} transitions across {len(clubs)} clubs.")
