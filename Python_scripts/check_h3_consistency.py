# ============================================================
# check_h3_consistency.py
#
# Verifica coherencia entre:
#  - Taxi_clusters_H3.csv
#  - h3_valid_neighbors_manhattan_normalized_neighbors.csv
#
# Genera:
#  - h3_only_in_taxi.csv
#  - h3_only_in_neighbors.csv
# ============================================================

import pandas as pd

# ----------------------------
# Rutas
# ----------------------------

taxi_path = "/Users/rgarrido/Documents/Investigacion/_Papers2025/Automatas-Percolation/Uber_NYC/quantum_behavior/Taxi_clusters_H3.csv"
neigh_path = "/Users/rgarrido/Documents/Investigacion/_Papers2025/Automatas-Percolation/Uber_NYC/data/h3_valid_neighbors_manhattan_normalized_neighbors.csv"

# ----------------------------
# Cargar datos
# ----------------------------

taxi = pd.read_csv(taxi_path)
neighbors = pd.read_csv(neigh_path)

# Normalizar nombres (asumimos que las dos primeras columnas son h3 y vecino)
neighbors = neighbors.rename(columns={neighbors.columns[0]: "h3_id",
                                      neighbors.columns[1]: "neighbor"})

# ----------------------------
# Conjuntos H3
# ----------------------------

h3_taxi = set(taxi["h3_id"].unique())
h3_neigh = set(neighbors["h3_id"].unique()) | set(neighbors["neighbor"].unique())

# ----------------------------
# Comparaciones
# ----------------------------

only_in_taxi = sorted(h3_taxi - h3_neigh)
only_in_neigh = sorted(h3_neigh - h3_taxi)
common = h3_taxi & h3_neigh

# ----------------------------
# Reporte
# ----------------------------

print("\n===== RESUMEN CONSISTENCIA H3 =====")
print(f"H3 en Taxi:      {len(h3_taxi)}")
print(f"H3 en Vecinos:   {len(h3_neigh)}")
print(f"Comunes:         {len(common)}")
print(f"Solo en Taxi:    {len(only_in_taxi)}")
print(f"Solo en Vecinos: {len(only_in_neigh)}")

# ----------------------------
# Guardar discrepancias
# ----------------------------

pd.DataFrame({"h3_id": only_in_taxi}).to_csv("h3_only_in_taxi.csv", index=False)
pd.DataFrame({"h3_id": only_in_neigh}).to_csv("h3_only_in_neighbors.csv", index=False)

print("\nArchivos generados:")
print(" - h3_only_in_taxi.csv")
print(" - h3_only_in_neighbors.csv\n")
