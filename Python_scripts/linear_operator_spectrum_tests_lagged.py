# ============================================================
# linear_operator_spectrum_tests_lagged.py
#
# Reconstruye operador A_Delta: x_{t+Delta} ~ A x_t
# usando bins existentes (no consecutivos)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EVENTS_PATH = "sparse_events_10min.csv"

DELTA = 12        # 6 bins = 1 hora (prueba 3,6,12)
RANK = 20
CENTER = True

# ----------------------------
# 1) Cargar y preparar
# ----------------------------
ev = pd.read_csv(EVENTS_PATH)
ev["t"] = ev["t"].astype(int)

hexes = sorted(ev["h3_id"].unique())
h2i = {h:i for i,h in enumerate(hexes)}
N = len(hexes)

groups = ev.groupby(["fecha","t"])["h3_id"].apply(list)
keys = sorted(groups.index.tolist())

# ----------------------------
# 2) Construir índice global
# ----------------------------
pos = {k:i for i,k in enumerate(keys)}
T = len(keys)

X = np.zeros((N, T), dtype=np.float32)
for col,(f,t) in enumerate(keys):
    for h in groups[(f,t)]:
        X[h2i[h], col] = 1.0

if CENTER:
    X = X - X.mean(axis=1, keepdims=True)

# ----------------------------
# 3) Armar pares con lag DELTA
# ----------------------------
cols1, cols2 = [], []
for i,(f,t) in enumerate(keys):
    k2 = (f, t+DELTA)
    if k2 in pos:
        cols1.append(i)
        cols2.append(pos[k2])

X1 = X[:, cols1]
X2 = X[:, cols2]

print(f"N hex: {N}")
print(f"Total bins: {T}")
print(f"Transiciones con Δ={DELTA}: {X1.shape[1]}")

if X1.shape[1] < 50:
    print("⚠️ Muy pocas transiciones. Prueba otro DELTA.")
    exit()

# ----------------------------
# 4) DMD reducido
# ----------------------------
U, s, Vt = np.linalg.svd(X1, full_matrices=False)
r = min(RANK, len(s))
U_r = U[:, :r]
s_r = s[:r]
V_r = Vt[:r, :].T

A_tilde = (U_r.T @ X2 @ V_r) @ np.diag(1.0 / s_r)

eigvals, _ = np.linalg.eig(A_tilde)

# ----------------------------
# 5) Diagnósticos
# ----------------------------
mag = np.abs(eigvals)
ang = np.angle(eigvals)

freq_cpd = (ang / (2*np.pi)) * (144.0/DELTA)

spec = pd.DataFrame({
    "eig_real": np.real(eigvals),
    "eig_imag": np.imag(eigvals),
    "abs": mag,
    "angle_rad": ang,
    "freq_cycles_per_day": freq_cpd
}).sort_values("abs", ascending=False)

spec.to_csv("koopman_eigs_lagged.csv", index=False)

near_unit = (mag > 0.95) & (mag < 1.05)
osc = near_unit & (np.abs(ang) > 1e-3)

Ainv = np.linalg.pinv(A_tilde)
I = np.eye(r)
rev_err = np.linalg.norm(A_tilde @ Ainv - I) / np.linalg.norm(I)
unit_err = np.linalg.norm(A_tilde.conj().T @ A_tilde - I) / np.linalg.norm(I)

with open("koopman_unitarity_test_lagged.txt", "w") as f:
    f.write("=== Operador lineal lagged ===\n")
    f.write(f"DELTA = {DELTA} bins (10 min c/u)\n")
    f.write(f"transiciones = {X1.shape[1]}\n")
    f.write(f"rank r = {r}\n")
    f.write(f"near-unit eigenvalues (0.95<|λ|<1.05): {int(np.sum(near_unit))}\n")
    f.write(f"oscillatory near-unit: {int(np.sum(osc))}\n")
    f.write(f"unitarity error ||A* A - I||/||I||: {unit_err:.6f}\n")
    f.write(f"reversibility error ||A A^+ - I||/||I||: {rev_err:.6f}\n")

print("Archivos creados:")
print(" - koopman_eigs_lagged.csv")
print(" - koopman_unitarity_test_lagged.txt")

# ----------------------------
# 6) Plot espectro
# ----------------------------
plt.figure(figsize=(6,6))
plt.scatter(np.real(eigvals), np.imag(eigvals), s=25)
theta = np.linspace(0, 2*np.pi, 400)
plt.plot(np.cos(theta), np.sin(theta), linewidth=1)
plt.title(f"Espectro complejo operador (Δ={DELTA})")
plt.xlabel("Re(λ)")
plt.ylabel("Im(λ)")
plt.axis("equal")
plt.tight_layout()
plt.savefig("koopman_spectrum_lagged.png", dpi=150)
plt.close()

print(" - koopman_spectrum_lagged.png")
