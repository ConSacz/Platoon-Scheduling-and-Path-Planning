import numpy as np
import os

from locations import locations

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------------------------------
# Build coordinate string
# --------------------------------------------------

coords = []

for info in locations.values():
    lat, lon = info["coord"]
    coords.append([lon, lat])

coords = np.array(coords)

# Euclidean distance matrix
distance= np.linalg.norm(
    coords[:, None, :] - coords[None, :, :],
    axis=2
)

np.save("data/distance_matrix.npy", distance)


print("Saved distance matrix:", distance.shape)
