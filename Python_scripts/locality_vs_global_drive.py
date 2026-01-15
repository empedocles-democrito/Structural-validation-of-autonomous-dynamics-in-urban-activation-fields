# ============================================================
# locality_vs_global_drive.py
#
# Compara modelos:
#   M0: solo tiempo
#   M1: tiempo + vecindario (t)
#   M2: tiempo + vecindario (t) + memoria (t-1) + self(t)
#
# Output:
#   model_comparison.csv
# ============================================================

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

events = pd.read_csv("sparse_events_10min.csv")  # fecha,t,h3_id,active=1
neighbors = pd.read_csv("h3_neighbors_active_only.csv")
neighbors = neighbors.rename(columns={neighbors.columns[0]:"h3_id", neighbors.columns[1]:"neighbor"})

# --- adjacency
adj = defaultdict(set)
for a,b in neighbors[["h3_id","neighbor"]].itertuples(index=False):
    adj[a].add(b); adj[b].add(a)

all_h3 = sorted(set(events["h3_id"].unique()))
all_h3_set = set(all_h3)

# --- index actives per (fecha,t)
actives = defaultdict(set)
for f,t,h in events[["fecha","t","h3_id"]].itertuples(index=False):
    actives[(f,int(t))].add(h)

def hour_feats(t):
    # t in 1..144
    ang = 2*np.pi*(t/144.0)
    return np.sin(ang), np.cos(ang)

rng = np.random.default_rng(123)

def sample_negatives(f, t, k=5):
    """Negativos: mezcla de vecinos de activos en t y uniformes"""
    A = actives.get((f,t), set())
    # pool cercano: vecinos de activos
    near = set()
    for h in A:
        near |= adj[h]
    near = list((near - A) & all_h3_set)

    negs = set()
    # mitad cerca si existe
    if len(near) > 0:
        take = min(len(near), k//2)
        negs |= set(rng.choice(near, size=take, replace=False))
    # resto uniforme
    while len(negs) < k:
        cand = rng.choice(all_h3)
        if cand not in A:
            negs.add(cand)
    return list(negs)

rows = []
for (f,t), A in actives.items():
    # saltar último bin del día
    if t >= 144:
        continue
    A_next = actives.get((f,t+1), set())

    # positivos = activos en t+1 (sparse, pero consistente)
    pos = list(A_next)
    # negativos: k por positivo (si no hay pos, sampleamos un poco igual para aprender M0)
    if len(pos)==0:
        # tomar algunos negativos para que exista fila
        neg = sample_negatives(f,t,k=10)
        candidates = [(h,0) for h in neg]
    else:
        candidates = [(h,1) for h in pos]
        for h in pos:
            for n in sample_negatives(f,t,k=5):
                candidates.append((n,0))

    for h,y in candidates:
        # features locales
        A_t = actives.get((f,t), set())
        A_tm1 = actives.get((f,t-1), set())
        neigh_t = sum((nb in A_t) for nb in adj[h])
        neigh_tm1 = sum((nb in A_tm1) for nb in adj[h])
        self_t = 1 if h in A_t else 0
        hs, hc = hour_feats(t+1)  # predice t+1
        rows.append((f,t+1,h,y,hs,hc,neigh_t,neigh_tm1,self_t))

df = pd.DataFrame(rows, columns=["fecha","t","h3_id","y","hour_sin","hour_cos","neigh_t","neigh_tm1","self_t"])

# --- split por fecha (evita leakage intra-día)
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=123)
train_idx, test_idx = next(gss.split(df, groups=df["fecha"]))
train, test = df.iloc[train_idx], df.iloc[test_idx]

def fit_eval(cols, name):
    Xtr, ytr = train[cols].values, train["y"].values
    Xte, yte = test[cols].values, test["y"].values

    clf = LogisticRegression(max_iter=200, n_jobs=1)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:,1]

    return {
        "model": name,
        "logloss": float(log_loss(yte, p)),
        "auc": float(roc_auc_score(yte, p)),
        "n_train": int(len(train)),
        "n_test": int(len(test))
    }

results = []
results.append(fit_eval(["hour_sin","hour_cos"], "M0_time_only"))
results.append(fit_eval(["hour_sin","hour_cos","neigh_t"], "M1_time_plus_local"))
results.append(fit_eval(["hour_sin","hour_cos","neigh_t","neigh_tm1","self_t"], "M2_time_local_memory"))

out = pd.DataFrame(results).sort_values("logloss")
out.to_csv("model_comparison.csv", index=False)

print(out)
print("\nArchivo creado: model_comparison.csv")
