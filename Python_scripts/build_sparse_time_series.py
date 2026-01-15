# ============================================================
# build_sparse_time_series.py
#
# Construye estados activos por intervalo de 10 min
# ============================================================

import pandas as pd
from datetime import datetime

taxi_path = "/Users/rgarrido/Documents/Investigacion/_Papers2025/Automatas-Percolation/Uber_NYC/quantum_behavior/Taxi_clusters_H3.csv"

df = pd.read_csv(taxi_path)

# ----------------------------
# Discretizar hora → bin 10 min
# ----------------------------

def time_to_bin(h):
    h = h.strip()
    dt = datetime.strptime(h, "%H:%M:%S")
    return (dt.hour * 60 + dt.minute) // 10 + 1

df["t"] = df["hora"].apply(time_to_bin)

# ----------------------------
# Dataset ralo
# ----------------------------

sparse = df[["fecha", "t", "h3_id"]].drop_duplicates()
sparse["active"] = 1

sparse = sparse.sort_values(["fecha", "t", "h3_id"])

sparse.to_csv("sparse_events_10min.csv", index=False)

print("Archivo creado:")
print("sparse_events_10min.csv")
print("Fechas:", sparse["fecha"].nunique())
print("Total eventos:", len(sparse))
