# ============================================================
# linear_operator_spectrum_tests.py
#
# 1) Construye matriz X_t (bins) x (hex) en forma densa moderada:
#    - 429 hex, 339*144 = 48816 bins máx (pero puede tener huecos)
# 2) Reconstruye operador A por DMD en subespacio (r reducido)
# 3) Espectro complejo y diagnósticos:
#    - histograma |lambda|
#    - ángulos (frecuencias)
#    - test oscilatorio: cuantos eigenvalores con fase !=0 y |lambda|~1
#    - reversibilidad aproximada: ||A A^{-1} - I|| en el subespacio (pseudo-inversa)
#
# Inputs:
#   - sparse_events_10min.csv
# Outputs:
#   - koopman_eigs.csv
#   - koopman_spectrum.png
#   - koopman_unitarity_test.txt
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EVENTS_PATH = "sparse_events_10min.csv"

# parámetros
RANK = 20          # dimensión del subespacio (ajusta 10-50)
CENTER = True      # centrar por media
MIN_BINS_PER_DAY = 100  # filtra días con pocos bins observados (robustez)

# ----------------------------
# 1) Cargar eventos y construir índices
# ----------------------------
ev = pd.read_csv(EVENTS_PATH)
ev["t"] = ev["t"].astype(int)

hexes = sorted(ev["h3_id"].unique())
h2i = {h:i for i,h in enumerate(hexes)}
N = len(hexes)

# Agrupar por fecha y t
groups = ev.groupby(["fecha","t"])["h3_id"].apply(list)

# ordenar bins globales por (fecha,t)
index = sorted(groups.index.tolist())

# filtrar días con pocos bins (opcional)
counts_by_day = pd.Series([f for f,_ in index]).value_counts()
valid_days = set(counts_by_day[counts_by_day >= MIN_BINS_PER_DAY].index)
index = [k for k in index if k[0] in valid_days]

T = len(index)
print(f"N hex: {N}, bins usados: {T}, días usados: {len(valid_days)}")

# construir X (N x T) para DMD estándar
X = np.zeros((N, T), dtype=np.float32)
for col,(f,t) in enumerate(index):
    for h in groups[(f,t)]:
        X[h2i[h], col] = 1.0

if CENTER:
    X = X - X.mean(axis=1, keepdims=True)

# ----------------------------
# 2) Construir pares (X1, X2) con bins consecutivos dentro del mismo día
# ----------------------------
# usamos solo transiciones (f,t)->(f,t+1) presentes
cols1 = []
cols2 = []
pos = {k:i for i,k in enumerate(index)}
for i,(f,t) in enumerate(index):
    k2 = (f, t+1)
    if k2 in pos:
        cols1.append(i)
        cols2.append(pos[k2])

X1 = X[:, cols1]
X2 = X[:, cols2]
print(f"Transiciones usadas: {X1.shape[1]}")

# ----------------------------
# 3) DMD en subespacio reducido
#    A_tilde = U^T X2 V Sigma^{-1}
# ----------------------------
U, s, Vt = np.linalg.svd(X1, full_matrices=False)
r = min(RANK, len(s))
U_r = U[:, :r]
s_r = s[:r]
V_r = Vt[:r, :].T

A_tilde = (U_r.T @ X2 @ V_r) @ np.diag(1.0 / s_r)

# eigen-decomp
eigvals, W = np.linalg.eig(A_tilde)

# ----------------------------
# 4) Diagnósticos espectrales
# ----------------------------
mag = np.abs(eigvals)
ang = np.angle(eigvals)  # en radianes por paso (10 min)

# frecuencia en ciclos por día: omega = ang/(2pi) por paso; pasos/día=144
freq_cpd = (ang / (2*np.pi)) * 144.0

spec = pd.DataFrame({
    "eig_real": np.real(eigvals),
    "eig_imag": np.imag(eigvals),
    "abs": mag,
    "angle_rad": ang,
    "freq_cycles_per_day": freq_cpd
}).sort_values("abs", ascending=False)

spec.to_csv("koopman_eigs.csv", index=False)

# ----------------------------
# 5) Test “cuasi-unitario” y oscilatorio
# ----------------------------
# umbrales
near_unit = (mag > 0.95) & (mag < 1.05)
osc = near_unit & (np.abs(ang) > 1e-3)

n_near = int(np.sum(near_unit))
n_osc = int(np.sum(osc))

# reversibilidad aproximada en subespacio: usando pseudo-inversa
Ainv = np.linalg.pinv(A_tilde)
I = np.eye(r)
rev_err = np.linalg.norm(A_tilde @ Ainv - I) / np.linalg.norm(I)

# “unitarity” aproximada: A^* A ~ I si fuera unitario (en subespacio)
unit_err = np.linalg.norm(A_tilde.conj().T @ A_tilde - I) / np.linalg.norm(I)

with open("koopman_unitarity_test.txt", "w") as f:
    f.write("=== Diagnóstico operador lineal (DMD/Koopman) ===\n")
    f.write(f"hex N = {N}\n")
    f.write(f"rank r = {r}\n")
    f.write(f"transiciones = {X1.shape[1]}\n")
    f.write(f"near-unit eigenvalues (0.95<|λ|<1.05): {n_near}\n")
    f.write(f"oscillatory near-unit (fase!=0): {n_osc}\n")
    f.write(f"unitarity error ||A* A - I||/||I||: {unit_err:.6f}\n")
    f.write(f"reversibility error ||A A^+ - I||/||I||: {rev_err:.6f}\n")

print("Archivos creados:")
print(" - koopman_eigs.csv")
print(" - koopman_unitarity_test.txt")

# ----------------------------
# 6) Plot espectro complejo
# ----------------------------
plt.figure(figsize=(6,6))
plt.scatter(np.real(eigvals), np.imag(eigvals), s=25)
theta = np.linspace(0, 2*np.pi, 400)
plt.plot(np.cos(theta), np.sin(theta), linewidth=1)  # círculo unitario
plt.title("Espectro complejo del operador (subespacio DMD)")
plt.xlabel("Re(λ)")
plt.ylabel("Im(λ)")
plt.axis("equal")
plt.tight_layout()
plt.savefig("koopman_spectrum.png", dpi=150)
plt.close()

print(" - koopman_spectrum.png")
