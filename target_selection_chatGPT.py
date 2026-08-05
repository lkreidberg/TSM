import numpy as np

# --- NumPy/Astropy compatibility shim ---
# NumPy 2.4+ removed np.in1d; some Astropy versions still try to access it at import time.
# Define it before importing astropy.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

import matplotlib.pyplot as plt
from astropy.io import ascii

d = ascii.read("NASAExoArchive_2026-03-02_aggregate.csv", format="csv")

# selection
ind = (d["pl_rade"] > 1.8) & (d["pl_bmasse"] < 17) & (d["TSM"] > 110)
d = d[ind]

ind = np.argsort(d["TSM"])
d = d[ind][::-1]

print("Planet Name\tMass(Earth)\tTSM")
for i in range(len(d)):
    print(d["pl_name"][i], "\t", d["pl_bmasse"][i], "\t", d["TSM"][i])
