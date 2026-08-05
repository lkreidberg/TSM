import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from pylab import hist

d = ascii.read("NASAExoArchive_2023-06-13_aggregate.csv", format = 'csv')

#best phase curves
"""ind = (d['pl_rade']<3.0)&(d['pl_rade'] > 1.8)&(d['ESM'] > 10)&(d['sy_kmag'] > 6.)&(d['pl_orbper'] < 2)
d = d[ind]

ind = np.argsort(d['ESM'])
d = d[ind]
d = d[::-1]"""

# best transits of hot sub-Neptunes
#ind = (d['pl_rade']>3.0)&(d['pl_rade'] < 4.0)&(d['TSM'] > 60)&(d['pl_Teq'] < 500)

nbins = 20

plt.subplot(131)
plt.title("Gas giants")
ind = (d['pl_rade']>8.0)&(d['pl_rade'] < 20.5)
hist(d['TSM'][ind], color = 'orange', bins = nbins)
plt.xlabel("TSM")
plt.ylabel("Frequency")

ind2 = np.argsort(d['TSM'][ind])
ind2 = ind2[::-1]
print(d[ind2]['pl_name'][ind2])

ind = d['pl_name'] == 'HD 209458 b'
print("HD209", d['TSM'][ind])

plt.subplot(132)
plt.title("Sub-Neptunes")
ind = (d['pl_rade']>2.0)&(d['pl_rade'] < 3.5)
hist(d['TSM'][ind], color = 'blue', bins = nbins)
plt.xlabel("TSM")
ind = d['pl_name'] == 'GJ 3470 b'
print("GJ 3470 b", d['TSM'][ind])

ind2 = np.argsort(d['TSM'][ind])
ind2 = ind2[::-1]
print(d[ind2]['pl_name'][ind2])

plt.subplot(133)
plt.title("Temperate terrestrials")
ind = (d['pl_rade']>0.5)&(d['pl_rade'] < 1.6)&(d['pl_Teq'] < 800)
hist(d['TSM'][ind], color = 'red', bins = nbins)
plt.xlim(0,120)
plt.xlabel("TSM")

ind = d['pl_name'] == 'TRAPPIST-1 b'
print("T1b TSM", d['TSM'][ind])

plt.show()
