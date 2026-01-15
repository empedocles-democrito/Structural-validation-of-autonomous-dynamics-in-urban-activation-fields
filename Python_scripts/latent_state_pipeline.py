# ============================================================
# latent_state_pipeline.py
#
# Estado latente a partir de activaciones ralas:
# 1) Construye X (T x N): T bins observados, N=429 hex
# 2) SVD (PCA) -> z_t (k dims)
# 3) Ajusta dinámica: z_{t+DELTA} = A z_t (ridge)
# 4) Evalúa:
#    - R2 en z (predicción en latente)
#    - R2 en x reconstruido (predicción en observables)
# 5) Exporta figuras y CSVs
#
# Inputs:
#   - sparse_events_10min.csv
#
# Outputs:
#   - latent_variance_explained.csv
#   - latent_states_z.csv
#   - latent_dynamics_A.npy
#   - latent_eval.txt
#   - fig_variance_explained.png
#   - fig_latent_phase.png
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EVENTS_PATH = "sparse_events_10min.csv"

K_MAX = 30        # calcula hasta 30 componentes para curva
K_USE = 6         # dimensión latente usada para dinámica (ajusta 2..10)
DELTA = 6         # lag en bins (6=1 hora). prueba 3,6,12
RIDGE = 1e-2      # regularización para A (ridge)
CENTER = True     # centrar columnas (hex) para PCA
SPLIT_BY_DATE = True
TEST_SIZE = 0.3
SEED = 123

# ----------------------------
# 1) Cargar y armar X (T x N)
# ----------------------------
ev = pd.read_csv(EVENTS_PATH)
ev["t"] = ev["t"].astype(int)

hexes = sorted(ev["h3_id"].unique())
h2i = {h:i for i,h in enumerate(hexes)}
N = len(hexes)

groups = ev.groupby(["fecha","t"])["h3_id"].apply(list)
keys = sorted(groups.index.tolist())
T = len(keys)

X = np.zeros((T, N), dtype=np.float32)
for r,(f,t) in enumerate(keys):
    for h in groups[(f,t)]:
        X[r, h2i[h]] = 1.0

# centrado por hex (quita baseline)
if CENTER:
    X = X - X.mean(axis=0, keepdims=True)

# ----------------------------
# 2) SVD (PCA)
# ----------------------------
# X = U S V^T, filas=tiempo
U, s, Vt = np.linalg.svd(X, full_matrices=False)

# varianza explicada por componente
var = s**2
var_ratio = var / var.sum()
ve = pd.DataFrame({
    "component": np.arange(1, len(s)+1),
    "singular_value": s,
    "var_ratio": var_ratio,
    "cum_var_ratio": np.cumsum(var_ratio)
})
ve.iloc[:K_MAX].to_csv("latent_variance_explained.csv", index=False)

# plot varianza explicada acumulada
plt.figure(figsize=(9,4))
plt.plot(ve["component"][:K_MAX], ve["cum_var_ratio"][:K_MAX], marker="o", linewidth=1)
plt.title("Varianza explicada acumulada (PCA/SVD)")
plt.xlabel("componentes")
plt.ylabel("varianza acumulada")
plt.tight_layout()
plt.savefig("fig_variance_explained.png", dpi=150)
plt.close()

# estado latente z_t = U_k * S_k (scores)
k = min(K_USE, U.shape[1])
Z = U[:, :k] * s[:k]  # (T x k)

# guardar z con índice temporal
z_df = pd.DataFrame(Z, columns=[f"z{i+1}" for i in range(k)])
z_df.insert(0, "fecha", [f for f,_ in keys])
z_df.insert(1, "t", [t for _,t in keys])
z_df.to_csv("latent_states_z.csv", index=False)

# ----------------------------
# 3) Construir pares con DELTA: z_{t+Δ} = A z_t
# ----------------------------
pos = {k:i for i,k in enumerate(keys)}
idx1, idx2 = [], []
for i,(f,t) in enumerate(keys):
    k2 = (f, t+DELTA)
    if k2 in pos:
        idx1.append(i)
        idx2.append(pos[k2])

Z1 = Z[idx1, :]
Z2 = Z[idx2, :]

# split train/test por fecha (recomendado)
if SPLIT_BY_DATE:
    rng = np.random.default_rng(SEED)
    fechas = np.array([keys[i][0] for i in idx1])
    uniq = np.unique(fechas)
    rng.shuffle(uniq)
    cut = int(len(uniq) * (1-TEST_SIZE))
    train_days = set(uniq[:cut])

    tr = np.array([i for i in range(len(idx1)) if fechas[i] in train_days])
    te = np.array([i for i in range(len(idx1)) if fechas[i] not in train_days])

    Z1_tr, Z2_tr = Z1[tr], Z2[tr]
    Z1_te, Z2_te = Z1[te], Z2[te]
else:
    # split simple
    n = Z1.shape[0]
    cut = int(n*(1-TEST_SIZE))
    Z1_tr, Z2_tr = Z1[:cut], Z2[:cut]
    Z1_te, Z2_te = Z1[cut:], Z2[cut:]

# ----------------------------
# 4) Estimar A con ridge: A = (Z1^T Z1 + λI)^{-1} Z1^T Z2
# ----------------------------
I = np.eye(k)
A = np.linalg.solve(Z1_tr.T @ Z1_tr + RIDGE*I, Z1_tr.T @ Z2_tr)  # (k x k)

np.save("latent_dynamics_A.npy", A)

# ----------------------------
# 5) Evaluación (R2 en latente)
# ----------------------------
def r2(y, yhat):
    ssr = np.sum((y - yhat)**2)
    sst = np.sum((y - y.mean(axis=0, keepdims=True))**2)
    return 1.0 - ssr/sst if sst > 0 else np.nan

Z2_hat_tr = Z1_tr @ A
Z2_hat_te = Z1_te @ A

r2_z_tr = r2(Z2_tr, Z2_hat_tr)
r2_z_te = r2(Z2_te, Z2_hat_te)

# ----------------------------
# 6) Evaluación en observables x (reconstrucción)
# X_hat(t+Δ) = Z_hat(t+Δ) V_k^T  (porque Z = U S)
V_k = Vt[:k, :]  # (k x N)

X2 = X[idx2, :]              # reales
X2_hat = (Z1 @ A) @ V_k      # predicción 1-step en Δ

# medir R2 global en x sobre las transiciones usadas
r2_x = r2(X2, X2_hat)

# ----------------------------
# 7) Gráfico “phase portrait” (z1 vs z2)
# ----------------------------
if k >= 2:
    plt.figure(figsize=(6,6))
    plt.scatter(Z[:,0], Z[:,1], s=6)
    plt.title("Retrato de fase latente (z1 vs z2)")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.tight_layout()
    plt.savefig("fig_latent_phase.png", dpi=150)
    plt.close()

# ----------------------------
# 8) Reporte
# ----------------------------
with open("latent_eval.txt", "w") as f:
    f.write("=== Estado latente + dinámica lineal ===\n")
    f.write(f"N hex = {N}\n")
    f.write(f"T bins = {T}\n")
    f.write(f"K_USE = {k}\n")
    f.write(f"DELTA = {DELTA} bins (10 min)\n")
    f.write(f"RIDGE = {RIDGE}\n")
    f.write(f"Transiciones usadas = {len(idx1)}\n")
    f.write(f"R2 latent (train) = {r2_z_tr:.6f}\n")
    f.write(f"R2 latent (test)  = {r2_z_te:.6f}\n")
    f.write(f"R2 observables X (all pairs) = {r2_x:.6f}\n")
    f.write("\nVarianza acumulada (primeros componentes):\n")
    for j in range(min(10, len(s))):
        f.write(f"  k={j+1}: cum_var={ve['cum_var_ratio'].iloc[j]:.6f}\n")

print("Listo. Archivos creados:")
print(" - latent_variance_explained.csv")
print(" - latent_states_z.csv")
print(" - latent_dynamics_A.npy")
print(" - latent_eval.txt")
print(" - fig_variance_explained.png")
print(" - fig_latent_phase.png (si k>=2)")
