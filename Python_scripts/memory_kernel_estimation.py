# ============================================================
# memory_kernel_estimation.py
#
# Entrena modelo logístico con memoria k=12 y extrae un "kernel"
# temporal a partir de coeficientes por lag:
#   - self_lagℓ
#   - neigh_lagℓ
#
# Grafica decaimiento y ajusta:
#   1) Exponencial:  c(ℓ) = A exp(-ℓ/τ)   -> τ (bins)
#   2) Potencia:     c(ℓ) = B (ℓ+1)^(-γ)  -> γ
#
# Inputs:
#   - sparse_events_10min.csv
#   - h3_neighbors_active_only.csv
#
# Outputs:
#   - memory_kernel_coeffs.csv
#   - memory_kernel_decay.png
#   - memory_kernel_fit.txt
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression

from scipy.optimize import curve_fit

EVENTS_PATH = "sparse_events_10min.csv"
NEIGH_PATH  = "h3_neighbors_active_only.csv"

K = 12
NEG_PER_POS = 5
RANDOM_SEED = 123

# ----------------------------
# Cargar datos
# ----------------------------
events = pd.read_csv(EVENTS_PATH)
neighbors = pd.read_csv(NEIGH_PATH)
neighbors = neighbors.rename(columns={neighbors.columns[0]:"h3_id",
                                      neighbors.columns[1]:"neighbor"})
events["t"] = events["t"].astype(int)

all_h3 = sorted(events["h3_id"].unique())
all_h3_set = set(all_h3)
rng = np.random.default_rng(RANDOM_SEED)

# ----------------------------
# Adyacencia (kring1)
# ----------------------------
adj = defaultdict(set)
for a,b in neighbors[["h3_id","neighbor"]].itertuples(index=False):
    if a in all_h3_set and b in all_h3_set:
        adj[a].add(b); adj[b].add(a)

# ----------------------------
# Activos por (fecha,t)
# ----------------------------
A = defaultdict(set)
for f,t,h in events[["fecha","t","h3_id"]].itertuples(index=False):
    A[(f,t)].add(h)

by_date = defaultdict(list)
for (f,t) in A.keys():
    by_date[f].append(t)
for f in by_date:
    by_date[f] = sorted(set(by_date[f]))

def hour_feats(t):
    ang = 2*np.pi*(t/144.0)
    return np.sin(ang), np.cos(ang)

def sample_near_negatives(f, t, k):
    At = A.get((f,t), set())
    near = set()
    for h in At:
        near |= adj[h]
    near = list((near - At) & all_h3_set)

    negs = set()
    if len(near) > 0:
        take = min(len(near), max(1, k//2))
        negs |= set(rng.choice(near, size=take, replace=False))
    while len(negs) < k:
        cand = rng.choice(all_h3)
        if cand not in At:
            negs.add(cand)
    return list(negs)

# ----------------------------
# Construir dataset (candidatos) para y=activo en t
# ----------------------------
rows = []
for f, ts in by_date.items():
    ts_set = set(ts)
    for t in ts:
        if t <= K:
            continue
        At = A.get((f,t), set())
        if len(At) == 0:
            continue

        pos = list(At)
        cand = [(h,1) for h in pos]
        for h in pos:
            for n in sample_near_negatives(f, t-1, NEG_PER_POS):
                cand.append((n,0))

        hs, hc = hour_feats(t)
        for h,y in cand:
            rows.append((f,t,h,y,hs,hc))

base = pd.DataFrame(rows, columns=["fecha","t","h3_id","y","hour_sin","hour_cos"])

# ----------------------------
# Agregar features de memoria hasta K (self + neigh por lag)
# ----------------------------
def add_memory_features(df, k):
    out = df.copy()

    cache_A = {}
    for (f,t) in set(zip(out["fecha"], out["t"])):
        for lag in range(0, k+1):
            tt = int(t) - lag
            cache_A[(f,tt)] = A.get((f,tt), set())

    # construir columnas con bucles controlados
    for lag in range(0, k+1):
        self_col = []
        neigh_col = []
        for f,t,h in out[["fecha","t","h3_id"]].itertuples(index=False):
            tt = int(t) - lag
            At = cache_A[(f,tt)]
            self_col.append(1 if h in At else 0)
            neigh_col.append(sum((nb in At) for nb in adj[h]))
        out[f"self_lag{lag}"] = self_col
        out[f"neigh_lag{lag}"] = neigh_col

    return out

df = add_memory_features(base, K)

# ----------------------------
# Split por fecha (entrena solo en train)
# ----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(df, groups=df["fecha"]))
train = df.iloc[train_idx].reset_index(drop=True)

# ----------------------------
# Entrenar modelo grande
# ----------------------------
cols = ["hour_sin","hour_cos"] + \
       [f"self_lag{lag}" for lag in range(0,K+1)] + \
       [f"neigh_lag{lag}" for lag in range(0,K+1)]

X = train[cols].values
y = train["y"].values

clf = LogisticRegression(max_iter=500, n_jobs=1)
clf.fit(X, y)

coef = pd.Series(clf.coef_.ravel(), index=cols)

# ----------------------------
# Extraer "kernel" por lag
# ----------------------------
lags = np.arange(0, K+1)

self_w  = np.array([coef[f"self_lag{l}"] for l in lags], dtype=float)
neigh_w = np.array([coef[f"neigh_lag{l}"] for l in lags], dtype=float)

# Usamos magnitud absoluta para decaimiento (signo puede ser interpretativo)
self_abs  = np.abs(self_w)
neigh_abs = np.abs(neigh_w)

out = pd.DataFrame({
    "lag": lags,
    "self_coef": self_w,
    "neigh_coef": neigh_w,
    "self_abs": self_abs,
    "neigh_abs": neigh_abs
})
out.to_csv("memory_kernel_coeffs.csv", index=False)

# ----------------------------
# Ajustes para tiempo característico
# ----------------------------
def exp_model(l, A, tau, c0):
    return A*np.exp(-l/tau) + c0

def pow_model(l, B, gamma, c0):
    return B*(l+1.0)**(-gamma) + c0

fit_lines = []

def fit_decay(yabs, name):
    # usar lags>0 para evitar degeneración
    l = lags[1:]
    yy = yabs[1:]

    # si todo ~0, no se ajusta
    if np.all(yy <= 1e-12):
        return {"name": name, "status":"no_signal"}

    # Exponencial
    try:
        p0 = [yy[0], 3.0, yy[-1]]
        popt_e, _ = curve_fit(exp_model, l, yy, p0=p0, maxfev=20000)
        A_e, tau_e, c0_e = popt_e
        pred_e = exp_model(l, *popt_e)
        sse_e = float(np.sum((yy - pred_e)**2))
    except Exception:
        A_e=tau_e=c0_e=np.nan
        sse_e=np.inf

    # Potencia
    try:
        p0 = [yy[0], 1.0, yy[-1]]
        popt_p, _ = curve_fit(pow_model, l, yy, p0=p0, maxfev=20000)
        B_p, g_p, c0_p = popt_p
        pred_p = pow_model(l, *popt_p)
        sse_p = float(np.sum((yy - pred_p)**2))
    except Exception:
        B_p=g_p=c0_p=np.nan
        sse_p=np.inf

    best = "exp" if sse_e < sse_p else "pow"
    return {
        "name": name,
        "status": "ok",
        "best": best,
        "exp_A": A_e, "exp_tau": tau_e, "exp_c0": c0_e, "exp_sse": sse_e,
        "pow_B": B_p, "pow_gamma": g_p, "pow_c0": c0_p, "pow_sse": sse_p
    }

fit_self = fit_decay(self_abs, "self_abs")
fit_neig = fit_decay(neigh_abs, "neigh_abs")

with open("memory_kernel_fit.txt","w") as f:
    f.write("=== Modelo entrenado (k=12) ===\n")
    f.write(f"N train: {len(train)}\n")
    f.write("\n--- Ajuste decaimiento SELF ---\n")
    for k,v in fit_self.items():
        f.write(f"{k}: {v}\n")
    f.write("\n--- Ajuste decaimiento NEIGH ---\n")
    for k,v in fit_neig.items():
        f.write(f"{k}: {v}\n")

print("Archivos creados:")
print(" - memory_kernel_coeffs.csv")
print(" - memory_kernel_decay.png")
print(" - memory_kernel_fit.txt")

# ----------------------------
# Gráfico decaimiento
# ----------------------------
plt.figure(figsize=(9,4))
plt.plot(lags, self_abs, marker="o", linewidth=1, label="|self coef|")
plt.plot(lags, neigh_abs, marker="o", linewidth=1, label="|neigh coef|")
plt.title("Decaimiento temporal (kernel efectivo) por lag")
plt.xlabel("lag (bins de 10 min hacia atrás)")
plt.ylabel("magnitud |coeficiente|")
plt.legend()
plt.tight_layout()
plt.savefig("memory_kernel_decay.png", dpi=150)
plt.close()
