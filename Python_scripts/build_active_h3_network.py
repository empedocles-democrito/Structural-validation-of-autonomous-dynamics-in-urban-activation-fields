# ============================================================
# build_active_h3_network.py
#
# Construye la red solo con hexágonos realmente usados
# ============================================================

import pandas as pd

taxi_path = "/Users/rgarrido/Documents/Investigacion/_Papers2025/Automatas-Percolation/Uber_NYC/quantum_behavior/Taxi_clusters_H3.csv"
neigh_path = "/Users/rgarrido/Documents/Investigacion/_Papers2025/Automatas-Percolation/Uber_NYC/data/h3_valid_neighbors_manhattan_normalized_neighbors.csv"

taxi = pd.read_csv(taxi_path)
neighbors = pd.read_csv(neigh_path)

neighbors = neighbors.rename(columns={neighbors.columns[0]: "h3_id",
                                      neighbors.columns[1]: "neighbor"})

taxi_h3 = set(taxi["h3_id"].unique())

neighbors_active = neighbors[
    neighbors["h3_id"].isin(taxi_h3) &
    neighbors["neighbor"].isin(taxi_h3)
].copy()

neighbors_active.to_csv("h3_neighbors_active_only.csv", index=False)

print("Red restringida creada:")
print("h3_neighbors_active_only.csv")
print("Hexágonos activos:", len(taxi_h3))
print("Aristas activas:", len(neighbors_active))
