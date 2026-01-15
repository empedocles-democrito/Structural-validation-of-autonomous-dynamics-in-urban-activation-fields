# ============================================================
# propagation_test.py
#
# Distancia mínima entre activaciones consecutivas
# ============================================================

import pandas as pd
import numpy as np
from collections import defaultdict

# ----------------------------
# Cargar datos
# ----------------------------

events = pd.read_csv("sparse_events_10min.csv")
neighbors = pd.read_csv("h3_neighbors_active_only.csv")

neighbors = neighbors.rename(columns={neighbors.columns[0]: "h3_id",
                                      neighbors.columns[1]: "neighbor"})

# ----------------------------
# Construir grafo (no ponderado)
# ----------------------------

adj = defaultdict(set)
for _, r in neighbors.iterrows():
    adj[r["h3_id"]].add(r["neighbor"])
    adj[r["neighbor"]].add(r["h3_id"])

# ----------------------------
# BFS distancias locales
# ----------------------------

def bfs_distances(start_nodes, max_depth=6):
    dist = {}
    frontier = set(start_nodes)
    for d in range(max_depth+1):
        for node in frontier:
            if node not in dist:
                dist[node] = d
        new_frontier = set()
        for node in frontier:
            new_frontier |= adj[node]
        frontier = new_frontier - set(dist.keys())
    return dist

# ----------------------------
# Índice por tiempo
# ----------------------------

events = events.sort_values(["fecha","t"])
groups = dict(tuple(events.groupby(["fecha","t"])))

records = []

for (fecha,t), curr in groups.items():
    prev_key = (fecha, t-1)
    if prev_key not in groups:
        continue

    prev_nodes = set(groups[prev_key]["h3_id"])
    curr_nodes = set(curr["h3_id"])

    if len(prev_nodes)==0 or len(curr_nodes)==0:
        continue

    dist_map = bfs_distances(prev_nodes, max_depth=6)

    for node in curr_nodes:
        d = dist_map.get(node, np.nan)
        records.append(d)

distances = pd.Series(records).dropna()
distances.to_csv("propagation_distances.csv", index=False)

print("Archivo creado: propagation_distances.csv")
print("Media:", distances.mean())
print("Proporción d=0:", (distances==0).mean())
print("Proporción d=1:", (distances==1).mean())
print("Proporción d>=3:", (distances>=3).mean())
