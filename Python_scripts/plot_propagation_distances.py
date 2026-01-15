import pandas as pd
import matplotlib.pyplot as plt

# leer columna única sin header
df = pd.read_csv("propagation_distances.csv", header=None, names=["d"])

plt.figure(figsize=(7,4))
plt.hist(df["d"], bins=range(0, int(df["d"].max())+2), density=True)
plt.xlabel("Minimal graph distance to previous active cells")
plt.ylabel("Probability")
plt.title("Propagation distance distribution")
plt.tight_layout()
plt.savefig("propagation_distances.png", dpi=150)
plt.close()

print("Figura creada: propagation_distances.png")
