import pickle
import numpy as np
import networkx as nx

from locations import locations

distance = np.load("data/distance_matrix.npy")

LG = nx.DiGraph()

# --------------------------------------
# Node Index
# --------------------------------------

names = list(locations.keys())

index = {
    name: i
    for i, name in enumerate(names)
}

# --------------------------------------
# Nodes
# --------------------------------------

for name, info in locations.items():

    lat, lon = info["coord"]

    LG.add_node(
        name,
        type=info["type"],
        latitude=lat,
        longitude=lon
    )

# --------------------------------------
# Automatic Edges
# --------------------------------------

MAX_DISTANCE = 235  # km
MIN_DISTANCE = 100  # km

for i, u in enumerate(names):

    for j, v in enumerate(names):

        # Không nối node với chính nó
        if i == j:
            continue

        d = distance[i, j]

        # Bỏ qua khoảng cách không hợp lệ
        if not np.isfinite(d):
            continue

        # Chỉ giữ edge < 1200 km
        if MIN_DISTANCE < d < MAX_DISTANCE:

            LG.add_edge(
                u,
                v,
                distance=d
            )

# --------------------------------------
# Save
# --------------------------------------

with open("logistics_graph.pkl", "wb") as f:
    pickle.dump(LG, f)

print()
print("Logistics graph created.")
print("Nodes :", LG.number_of_nodes())
print("Edges :", LG.number_of_edges())