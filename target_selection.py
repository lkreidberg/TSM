import numpy as np
# --- NumPy/Astropy compatibility shim ---
# NumPy 2.4+ removed np.in1d; some Astropy versions still try to access it at import time.
# Define it before importing astropy.
if not hasattr(np, "in1d"):
    np.in1d = np.isin
import matplotlib.pyplot as plt
from astropy.io import ascii

d = ascii.read("NASAExoArchive_2026-03-02_aggregate.csv", format = 'csv')

#best phase curves
"""ind = (d['pl_rade']<3.0)&(d['pl_rade'] > 1.8)&(d['ESM'] > 10)&(d['sy_kmag'] > 6.)&(d['pl_orbper'] < 2)
d = d[ind]

ind = np.argsort(d['ESM'])
d = d[ind]
d = d[::-1]"""

# selection 
#ind = (d['pl_rade']>1.8)&(d['pl_rade']<2)&(d['pl_orbper'] < 1.5)&(d['ESM'] > 3)
#ind = (d['pl_rade']>1.8)&(d['pl_rade']<4)&(d['ESM'] > 3)
#ind = (d['pl_bmasse']<20.0)&(d['TSM'] > 70)&(d['st_teff']>5000)&(d['pl_rade']>2)
#ind = (d['pl_rade']>2.0)&(d['pl_rade'] < 18.0)&(d['TSM'] > 40)&(d['pl_Teq'] < 300)#&(d['pl_Teq'] < 1650)
#ind = (d['TSM']>150)&(d['pl_rade']<8)
#ind = (d['pl_rade']>1.8)&(d['pl_rade']<4.5)&(d['st_rad']<0.4)&(d['TSM']>40)
#ind = (d['pl_rade']>5)&(d['pl_rade']<20)&(d['pl_orbper']<1.5)&(d['pl_Teq']>1900)
#ind = (d['pl_bmasse']>20)&(d['pl_bmasse']<70)&(d['pl_Teq']<1400)&(d['pl_Teq']>1100)
ind = (d['pl_rade']>1.8)&(d['pl_bmasse']<17)&(d['TSM']>110)

d = d[ind]

ind = np.argsort(d['TSM'])
d = d[ind]
d = d[::-1]

#print("Planet Name\t\t Rp\t Mp\t Mp/Rp**2\t Teq\t TSM\t ESM")
#for i in range(len(d)):
#    print(d['pl_name'][i], '\t', '{0:0.3f}'.format(d['pl_rade'][i]), '\t', '{0:0.1f}'.format(d['pl_bmasse'][i]), '\t\t', '{0:0.1f}'.format(d['pl_bmasse'][i]/(d['pl_rade'][i])**3), '\t', '{0:0.1f}'.format(d['pl_Teq'][i]), '\t', '{0:0.1f}'.format(d['TSM'][i]),'\t', '{0:0.1f}'.format(d['ESM'][i]))

print("Planet Name\tTSM")
for i in range(len(d)):
    print(d['pl_name'][i], '\t', d['pl_bmasse'][i], '\t', d['TSM'][i])


