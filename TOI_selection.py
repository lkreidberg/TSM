import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii

d = ascii.read("TOI2023-08-17_aggregate.csv", format = 'csv')


# selection 
ind = (d['pl_rade']>1.6)&(d['pl_rade']<6)&(d['TSM'] > 100)

d = d[ind]

ind = np.argsort(d['TSM'])
d = d[ind]
d = d[::-1]

print("Planet Name\t\t Rp\t Mp\t  Teq\t TSM")
for i in range(len(d)):
    print("TOI ", d['toi'][i], '\t', '{0:0.3f}'.format(d['pl_rade'][i]), '\t', '{0:0.1f}'.format(d['pl_bmasse'][i]), '\t\t',  '{0:0.1f}'.format(d['pl_eqt'][i]), '\t', '{0:0.1f}'.format(d['TSM'][i]))


