# ============================================================
# radial_scaling_test.py
#
# Test de escalamiento radial en distancia de grafo:
#   R(tau) = E[ min_{j in A_t} d(i,j)^2 | i in A_{t+tau} ]
#
# Inputs (en el mismo directorio):
#   - sparse_events_10min.csv
#   - h3_neighbors_active_only.csv
#
# Outputs:
#   - radial_scaling_R_tau.csv
#   - radial_scaling_fit.txt
#   - radial_scaling_Rtau.png
#   - radial_scaling_loglog.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque

EVENTS_PATH = "sparse_events_10min.csv"
NEIGH_PATH  = "h3_neighbors_active_only.csv"

MAX_TAU = 36          # 36*10min = 6 horas (ajusta si quieres)
USE_NULL = True       # nulo por aleatorización (recomendado)
NULL_SAMPLES = 5      # repeticiones por (fecha,t,tau) para baseline

# ----------------------------
# 1) Cargar datos
# ----------------------------
events = pd.read_csv(EVENTS_PATH)
neighbors = pd.read_csv(NEIGH_PATH)

neighbors = neighbors.rename(columns={neighbors.columns[0]: "h3_id",
                                      neighbors.columns[1]: "neighbor"})

# universo de nodos
nodes = sorted(set(events["h3_id"].unique()))
node_index = {h:i for i,h in enumerate(nodes)}
N = len(nodes)

# ----------------------------
# 2) Adyacencia
# ----------------------------
adj = defaultdict(list)
for a,b in neighbors[["h3_id","neighbor"]].itertuples(index=False):
    if a in node_index and b in node_index:
        adj[a].append(b)
        adj[b].append(a)

# ----------------------------
# 3) Precomputar distancias all-pairs (BFS por nodo)
# ----------------------------
dist = np.full((N, N), np.inf, dtype=np.float32)

for h in nodes:
    s = node_index[h]
    dist[s, s] = 0.0
    q = deque([h])
    seen = {h}
    while q:
        u = q.popleft()
        du = dist[s, node_index[u]]
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                dist[s, node_index[v]] = du + 1.0
                q.append(v)

# chequeo conectividad
if np.isinf(dist).any():
    print("⚠️ Advertencia: hay pares de nodos desconectados en la red inducida.")

# ----------------------------
# 4) Índice de activos A_(fecha,t)
# ----------------------------
events["t"] = events["t"].astype(int)
actives = defaultdict(list)
for f,t,h in events[["fecha","t","h3_id"]].itertuples(index=False):
    actives[(f,t)].append(node_index[h])

# lista de keys por fecha para iterar eficiente
by_date = defaultdict(list)
for (f,t) in actives.keys():
    by_date[f].append(t)
for f in by_date:
    by_date[f] = sorted(by_date[f])

rng = np.random.default_rng(123)

# ----------------------------
# 5) Acumular R(tau)
# ----------------------------
sum_R = np.zeros(MAX_TAU+1, dtype=np.float64)
cnt_R = np.zeros(MAX_TAU+1, dtype=np.int64)

sum_R0 = np.zeros(MAX_TAU+1, dtype=np.float64)  # nulo
cnt_R0 = np.zeros(MAX_TAU+1, dtype=np.int64)

for f, ts in by_date.items():
    ts_set = set(ts)
    for t in ts:
        A_t = actives.get((f,t), [])
        if len(A_t) == 0:
            continue

        # min-dist a cada nodo desde el conjunto A_t
        # (min sobre filas dist[A_t, :])
        mind = np.min(dist[A_t, :], axis=0)

        for tau in range(1, MAX_TAU+1):
            t2 = t + tau
            if t2 not in ts_set:
                continue
            A_t2 = actives.get((f,t2), [])
            if len(A_t2) == 0:
                continue

            d2 = mind[A_t2]  # dist mínima para cada activo en t+tau
            R = np.mean(d2.astype(np.float64)**2)

            sum_R[tau] += R
            cnt_R[tau] += 1

            # nulo: mismos tamaños, nodos aleatorios (sin estructura espacial)
            if USE_NULL:
                k = len(A_t2)
                for _ in range(NULL_SAMPLES):
                    sample = rng.choice(N, size=k, replace=False)
                    R0 = np.mean((mind[sample].astype(np.float64))**2)
                    sum_R0[tau] += R0
                    cnt_R0[tau] += 1

# ----------------------------
# 6) Resultados + ajustes
# ----------------------------
taus = np.arange(1, MAX_TAU+1)
R_tau = np.array([sum_R[t]/cnt_R[t] if cnt_R[t]>0 else np.nan for t in taus], dtype=float)

out = pd.DataFrame({
    "tau": taus,
    "R_tau": R_tau,
    "n_pairs": [cnt_R[t] for t in taus]
})

if USE_NULL:
    R0_tau = np.array([sum_R0[t]/cnt_R0[t] if cnt_R0[t]>0 else np.nan for t in taus], dtype=float)
    out["R0_tau_null"] = R0_tau
    out["R_minus_null"] = out["R_tau"] - out["R0_tau_null"]

out.to_csv("radial_scaling_R_tau.csv", index=False)

# Ajuste de exponente: log R = a + alpha log tau (usando solo taus válidos)
mask = np.isfinite(R_tau) & (R_tau > 0)
x = np.log(taus[mask])
y = np.log(R_tau[mask])
alpha, a = np.polyfit(x, y, 1)

with open("radial_scaling_fit.txt", "w") as f:
    f.write(f"Fit log-log: log R = a + alpha log tau\n")
    f.write(f"alpha = {alpha:.4f}\n")
    f.write(f"a     = {a:.4f}\n")
    f.write(f"MAX_TAU = {MAX_TAU}\n")
    f.write(f"USE_NULL = {USE_NULL}, NULL_SAMPLES = {NULL_SAMPLES}\n")

print("Archivos creados:")
print(" - radial_scaling_R_tau.csv")
print(" - radial_scaling_fit.txt")

# ----------------------------
# 7) Gráficos
# ----------------------------
plt.figure(figsize=(9,4))
plt.plot(out["tau"], out["R_tau"], marker="o", linewidth=1)
if USE_NULL:
    plt.plot(out["tau"], out["R0_tau_null"], marker="o", linewidth=1)
plt.title("Escalamiento radial: R(tau) en distancia de grafo")
plt.xlabel("tau (bins de 10 min)")
plt.ylabel("R(tau) = E[min d^2]")
plt.tight_layout()
plt.savefig("radial_scaling_Rtau.png", dpi=150)
plt.close()

plt.figure(figsize=(9,4))
plt.loglog(out["tau"], out["R_tau"], marker="o", linewidth=1)
plt.title("Escalamiento radial (log-log)")
plt.xlabel("log tau")
plt.ylabel("log R(tau)")
plt.tight_layout()
plt.savefig("radial_scaling_loglog.png", dpi=150)
plt.close()

print(" - radial_scaling_Rtau.png")
print(" - radial_scaling_loglog.png")
