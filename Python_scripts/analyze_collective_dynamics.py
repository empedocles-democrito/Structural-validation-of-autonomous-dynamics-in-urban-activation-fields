# ============================================================
# analyze_collective_dynamics.py
#
# Magnetización proxy, autocorrelación y espectro
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram

path = "sparse_events_10min.csv"
df = pd.read_csv(path)

# ----------------------------
# Contar activos por (fecha,t)
# ----------------------------

mag = df.groupby(["fecha", "t"]).size().reset_index(name="m")
mag = mag.sort_values(["fecha", "t"]).reset_index(drop=True)

# ----------------------------
# Serie temporal continua
# ----------------------------

m = mag["m"].values
m = m - m.mean()

# ----------------------------
# Autocorrelación
# ----------------------------

def autocorr(x, max_lag):
    x = x - x.mean()
    result = np.correlate(x, x, mode='full')
    ac = result[result.size//2:]
    return ac[:max_lag] / ac[0]

acf = autocorr(m, 300)

plt.figure(figsize=(9,4))
plt.plot(acf)
plt.title("Autocorrelación magnetización proxy")
plt.xlabel("lag (10 min)")
plt.ylabel("ACF")
plt.tight_layout()
plt.savefig("acf_magnetization.png", dpi=150)
plt.close()

# ----------------------------
# Espectro de potencia
# ----------------------------

freqs, psd = periodogram(m, fs=1)

plt.figure(figsize=(9,4))
plt.semilogy(freqs, psd)
plt.title("Espectro de potencia magnetización proxy")
plt.xlabel("frecuencia (1/10min)")
plt.ylabel("PSD")
plt.tight_layout()
plt.savefig("psd_magnetization.png", dpi=150)
plt.close()

# ----------------------------
# Guardar serie
# ----------------------------

mag.to_csv("magnetization_timeseries.csv", index=False)

print("Archivos creados:")
print("acf_magnetization.png")
print("psd_magnetization.png")
print("magnetization_timeseries.csv")
