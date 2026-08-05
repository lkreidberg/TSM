#!/usr/bin/env python
# coding: utf-8

# # TSM Calculator
# 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from astropy import constants as const
from astroquery.mast import MastMissions
from astroquery.mast import Catalogs

def Planck_ratio(depth, Ts, Tp, lam):
    h = const.h.value
    c = const.c.value
    kB = const.k_B.value
    res = depth * 1e6 * (np.exp(h * c / (lam * kB * Ts)) - 1)/ (np.exp(h * c / (lam * kB * Tp)) - 1)
    return res

Rjup = const.R_jup/const.R_earth
Mjup = const.M_jup/const.M_earth
ReRs = const.R_earth/const.R_sun
MeMs = const.M_earth/const.M_sun

# ## Read in datasets

df_filename = "TOI_2023.07.30_07.46.29.csv"

df = pd.read_csv(
    df_filename,
    header=0
)

#guesses masses
df['pl_bmasse'] = 2.7*df['pl_rade']**1.3
df['pl_trandep'] = df['pl_trandep']/1e6


#scale factor for TSM calculation https://arxiv.org/pdf/1805.03671.pdf
df['scale'] = 0
df.loc[df['pl_rade'] <= 1.5, 'scale'] = 0.19
df.loc[df['pl_rade'] > 4, 'scale'] = 1.15
df.loc[(df['pl_rade'] <= 4)&(df['pl_rade'] >2.75), 'scale'] = 1.28
df.loc[(df['pl_rade'] <= 2.75)&(df['pl_rade'] > 1.5), 'scale'] = 1.26

""""df['TSMwrong'] = df['pl_rade']*df['pl_trandep']/(ReRs**2)*df['pl_eqt']/df['pl_bmasse']*10.**(-0.2*df['st_tmag'])*df['scale']

df = df.dropna(axis=0, subset='TSMwrong')
print("new df length", len(df))

df = df.sort_values(by=['TSMwrong'], ascending=False)"""

""""print("median Tmag = ", np.median(df['st_tmag']))
print("median TSM= ", np.median(df['TSMwrong']))

plt.hist(df['TSMwrong'], bins =50, range = (0,500))
plt.show()"""


df['Vmag'] = 0
df['Jmag'] = 0
df['Hmag'] = 0


#add additional columns from astroquery
for index, row in df.iterrows():
    tid = "TIC " + str(row['tid'])
    catalog_data = Catalogs.query_object(tid, catalog='TIC', radius=0.0001, row_limit =1)
    catalog_data = catalog_data[0]
    #print(catalog_data)
    df.loc[df.index[index], 'Vmag'] = catalog_data['Vmag']
    df.loc[df.index[index], 'Hmag'] = catalog_data['Hmag']
    df.loc[df.index[index], 'Jmag'] = catalog_data['Jmag']
    print(index)

    
# calculate TSM
df['TSM'] = df['pl_rade']*df['pl_trandep']/(ReRs**2)*df['pl_eqt']/df['pl_bmasse']*10.**(-0.2*df['Jmag'])*df['scale']

###############################################################
# save data frame

df.to_csv("TOI" + datetime.today().strftime("%Y-%m-%d") + "_aggregate.csv", index=False)

