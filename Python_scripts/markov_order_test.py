# ============================================================
# markov_order_test.py
#
# Test formal de no-Markovianidad por orden k:
# Compara modelos con memoria hasta k bins (10 min c/u)
#
# Modelos: k = 0..K
# Features:
#   - hora_sin, hora_cos
#   - neigh_{t-l}, self_{t-l} para l=0..k
#
# Métricas:
#   - logloss out-of-sample (split por fecha)
#   - AUC out-of-sample
#   - delta_logloss vs k=0
#
# Outputs:
#   - markov_order_comparison.csv
#   - markov_order_fit.png
#   - markov_order_best_model.txt
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

EVENTS_PATH = "sparse_events_10min.csv"
NEIGH_PATH  = "h3_neighbors_active_only.csv"

K_MAX = 12          # 12 bins = 2 horas (sube a 24 si quieres 4 horas)
NEG_PER_POS = 5     # negativos por positivo
RANDOM_SEED = 123

# ----------------------------
# Cargar datos
# ----------------------------
events = pd.read_csv(EVENTS_PATH)
neighbors = pd.read_csv(NEIGH_PATH)
neighbors = neighbors.rename(columns={neighbors.columns[0]:"h3_id",
                                      neighbors.columns[1]:"neighbor"})

events["t"] = events["t"].astype(int)

# Universo de hex observados
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
# Activos por (fecha,t) en set
# ----------------------------
A = defaultdict(set)
for f,t,h in events[["fecha","t","h3_id"]].itertuples(index=False):
    A[(f,t)].add(h)

# Lista de fechas y bins presentes
by_date = defaultdict(list)
for (f,t) in A.keys():
    by_date[f].append(t)
for f in by_date:
    by_date[f] = sorted(set(by_date[f]))

def hour_feats(t):
    ang = 2*np.pi*(t/144.0)
    return np.sin(ang), np.cos(ang)

def sample_near_negatives(f, t, k):
    """Negativos: preferir pool cercano (vecinos de activos en t), si no, uniforme."""
    At = A.get((f,t), set())
    near = set()
    for h in At:
        near |= adj[h]
    near = list((near - At) & all_h3_set)

    negs = set()
    # mitad cerca si hay
    if len(near) > 0:
        take = min(len(near), max(1, k//2))
        negs |= set(rng.choice(near, size=take, replace=False))
    # resto uniforme
    while len(negs) < k:
        cand = rng.choice(all_h3)
        if cand not in At:
            negs.add(cand)
    return list(negs)

# ----------------------------
# Construir dataset base (candidatos) una vez
# Target: activo en (t)
# Usamos como "t" el tiempo objetivo; features miran hacia atrás.
# ----------------------------
rows = []
for f, ts in by_date.items():
    ts_set = set(ts)
    for t in ts:
        if t <= K_MAX:  # no hay historia suficiente
            continue

        At = A.get((f,t), set())
        if len(At) == 0:
            continue

        # positivos = activos en t
        pos = list(At)
        cand = [(h,1) for h in pos]

        # negativos por positivo (muestreo desde tiempo t-1 para evitar leakage trivial)
        for h in pos:
            for n in sample_near_negatives(f, t-1, NEG_PER_POS):
                cand.append((n,0))

        hs, hc = hour_feats(t)
        for h,y in cand:
            rows.append((f,t,h,y,hs,hc))

base = pd.DataFrame(rows, columns=["fecha","t","h3_id","y","hour_sin","hour_cos"])

# ----------------------------
# Función para agregar features de memoria hasta k
# ----------------------------
def add_memory_features(df, k):
    out = df.copy()

    # precompute para cada fila (f,t,h) las variables self_{t-l}, neigh_{t-l}
    self_feats = {}
    neigh_feats = {}

    # cache por (f,t) para acelerar
    cache_A = {}
    for (f,t) in set(zip(out["fecha"], out["t"])):
        cache_A[(f,t)] = A.get((f,int(t)), set())

    for idx, r in out.iterrows():
        f = r["fecha"]; t = int(r["t"]); h = r["h3_id"]
        for lag in range(0, k+1):
            tt = t - lag
            At = cache_A.get((f,tt), set())
            self_feats[(idx,lag)] = 1 if h in At else 0
            neigh_feats[(idx,lag)] = sum((nb in At) for nb in adj[h])

    for lag in range(0, k+1):
        out[f"self_lag{lag}"] = [self_feats[(i,lag)] for i in out.index]
        out[f"neigh_lag{lag}"] = [neigh_feats[(i,lag)] for i in out.index]

    return out

# ----------------------------
# Split por fecha (generalización temporal)
# ----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(base, groups=base["fecha"]))
base_train = base.iloc[train_idx].reset_index(drop=True)
base_test  = base.iloc[test_idx].reset_index(drop=True)

def fit_eval_k(k):
    tr = add_memory_features(base_train, k)
    te = add_memory_features(base_test, k)

    cols = ["hour_sin","hour_cos"] + \
           [f"self_lag{lag}" for lag in range(0,k+1)] + \
           [f"neigh_lag{lag}" for lag in range(0,k+1)]

    Xtr, ytr = tr[cols].values, tr["y"].values
    Xte, yte = te[cols].values, te["y"].values

    clf = LogisticRegression(max_iter=300, n_jobs=1)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:,1]

    return {
        "k": k,
        "logloss": float(log_loss(yte, p)),
        "auc": float(roc_auc_score(yte, p)),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_features": int(len(cols))
    }

results = []
for k in range(0, K_MAX+1):
    results.append(fit_eval_k(k))

res = pd.DataFrame(results)
res["delta_logloss_vs_k0"] = res["logloss"] - float(res.loc[res["k"]==0,"logloss"])
res.to_csv("markov_order_comparison.csv", index=False)

best = res.sort_values("logloss").iloc[0]

with open("markov_order_best_model.txt","w") as f:
    f.write("Mejor k por logloss (menor es mejor)\n")
    f.write(best.to_string(index=False))
    f.write("\n")

print(res[["k","logloss","auc","delta_logloss_vs_k0","n_features"]])
print("\nArchivos creados:")
print(" - markov_order_comparison.csv")
print(" - markov_order_fit.png")
print(" - markov_order_best_model.txt")

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(9,4))
plt.plot(res["k"], res["logloss"], marker="o", linewidth=1)
plt.title("Test de orden Markov: logloss vs k (memoria)")
plt.xlabel("k (bins de 10 min hacia atrás)")
plt.ylabel("logloss (out-of-sample)")
plt.tight_layout()
plt.savefig("markov_order_fit.png", dpi=150)
plt.close()
