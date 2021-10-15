import numpy as np
from astropy.io import ascii

d = ascii.read("NASAExoArchive_2021-03-07_aggregate.csv", format= 'csv')
d['TSM'].fill_value = -999
d = d.filled()

ind = np.argsort(d['TSM'])
d = d[ind][::-1]

for i in range(len(d)):
#    if(d['TSM'][i]>92):
    if(d['TSM'][i]>30):
        if d['pl_T'][i] < 350.:
            print(d['pl_name'][i], d['TSM'][i], d['pl_bmasse'][i], d['pl_T'][i])

