#!/usr/bin/env python
# coding: utf-8

# # TSM Calculator
# 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from astropy import constants as const

Rjup = const.R_jup/const.R_earth
Mjup = const.M_jup/const.M_earth
ReRs = const.R_earth/const.R_sun
MeMs = const.M_earth/const.M_sun
# ## Read in datasets

df0_filename = "default_params0.csv"
df1_filename = "default_params1.csv"


df0 = pd.read_csv(
    df0_filename,
    header=0
).iloc[:, 1:]


#default parameter set = 1

df1 = pd.read_csv(
    df1_filename,
    header=0
).iloc[:, 1:]

#concatenates data sets
df = pd.concat([df0,df1])


# ## Convert the pubdate to a datetime object
# This is messy since there are three different datetime formats here.
# You must make sure that you sort before filtering.

df['dt_obj'] = df['pl_pubdate']
df['dt_obj'] = pd.to_datetime(df['pl_pubdate'], format="%Y-%m", errors='ignore')
df['dt_obj'] = pd.to_datetime(df['pl_pubdate'], format="%Y-%m-%d", errors='ignore')
df['dt_obj'] = pd.to_datetime(df['pl_pubdate'], format="%Y-%m-%d %H:%M", errors='ignore')
df['dt_obj'] = pd.to_datetime(df['pl_pubdate'], format="%Y-%m", errors='raise')

df = df.sort_values(by='dt_obj', ascending=False)

# ## Build up a filter to kill off unwanted results
# Finding the Stassun paper
df.loc[df['pl_refname'].str.contains("STASSUN"), 'pl_refname'].unique()

data_filter = (
    (df['pl_refname'] != '<a refstr=STASSUN_ET_AL__2017 href=https://ui.adsabs.harvard.edu/abs/2017AJ....153..136S/abstract target=ref>Stassun et al. 2017</a>')
)

# ## Groupby planet name and grab the most recent record

cols = df.columns.to_list()[1:]
agg_dict = dict(zip(cols, ['first'] * len(cols)))

#aggregate
df = df[data_filter].groupby('pl_name', as_index = False).agg(agg_dict)



#fill in columns where mass or radius are only in Jupiter units
df.pl_rade.fillna(df.pl_radj*Rjup, inplace=True)
df.pl_bmasse.fillna(df.pl_bmassj*Mjup, inplace=True)





#initialize new column for aggregate temperature
nplanets = len(df)

#first populate with insolation
df['pl_Teq'] = 278.*(df['pl_insol'])**0.25
#if insolation is unavailable, try equilibrium temperature
df.pl_Teq.fillna(df.pl_eqt, inplace=True)
#if equilibrium temperature is unavailable, calculate from a/Rs and Teff
df.pl_Teq.fillna(1/(np.sqrt(2*df['pl_ratdor']))*df['st_teff'], inplace=True)
#if a/Rs is unavailable, calculate it from a [AU] and Rs [Rsun]
df.pl_Teq.fillna(1/(np.sqrt(2*215.*df['pl_orbsmax']/df['st_rad']))*df['st_teff'], inplace=True)

#fill rprs if not given
df['pl_ratror'] = ReRs*df['pl_rade']/df['st_rad']
#df.pl_ratror.fillna(ReRs*df['pl_rade']/df['st_rad'], inplace=True)
#initialize new column for (Rp/Rs)**2
df['pl_rprs2'] = df['pl_ratror']**2

#scale factor for TSM calculation https://arxiv.org/pdf/1805.03671.pdf
df['scale'] = 0
df.loc[df['pl_rade'] <= 1.5, 'scale'] = 0.19
df.loc[df['pl_rade'] > 4, 'scale'] = 1.15
df.loc[(df['pl_rade'] <= 4)&(df['pl_rade'] >2.75), 'scale'] = 1.28
df.loc[(df['pl_rade'] <= 2.75)&(df['pl_rade'] > 1.5), 'scale'] = 1.26

#initialize new column for TSM
df['TSM'] = df['pl_rade']*df['pl_rprs2']/(ReRs**2)*df['pl_Teq']/df['pl_bmasse']*10.**(-0.2*df['sy_jmag'])*df['scale']

df['efficiency'] = 1
df.loc[df['sy_jmag'] <= 8.8, 'efficiency'] = (1.375*df['sy_jmag']**2 - 1.214*df['sy_jmag'] - 26.68)/64.58
df.loc[df['sy_jmag'] <= 5., 'efficiency'] = 0.3

df['efficiency_kmag'] = 1
df.loc[df['sy_kmag'] <= 8.8, 'efficiency_kmag'] = (1.375*df['sy_kmag']**2 - 1.214*df['sy_kmag'] - 26.68)/64.58
df.loc[df['sy_kmag'] <= 5., 'efficiency_kmag'] = 0.3
#correct TSM for observatinoal efficiency with HST/WFC3
#df['TSM'] = df['TSM']*np.sqrt(df['efficiency'])

noise_ref = 122 #photon noise wasp-103 122 ppm per 103 second exposure
kmag_ref = 10.767
df['noise'] = noise_ref *(10**(-0.4*(kmag_ref-df['sy_kmag'])))**(1/2)*(1/df['efficiency_kmag'])**(1/2)


h = const.h.value
c = const.c.value
kB = const.k_B.value


def Planck_ratio(depth, Ts, Tp, lam):
    res = depth * 1e6 * (np.exp(h * c / (lam * kB * Ts)) - 1)/ (np.exp(h * c / (lam * kB * Tp)) - 1)
    return res

# Fill ars if missing: a(AU)/Rs(Ro)*215
df.pl_ratdor.fillna(df['pl_orbsmax']/df['st_rad']*215, inplace=True)

# Fill insolation if missing: Ts^4/ars^2 * (215^2/5772^4) = Ts^4/ars^2 * 4.166e-11
df.pl_insol.fillna(df['st_teff']**4/df['pl_ratdor']**2*4.166e-11, inplace=True)

# initialize new columns with substellar temp., bare rock temperature
df['pl_Tsubst'] = df['pl_Teq']*np.sqrt(2) #df['st_teff']/np.sqrt(df['pl_ratdor'])
df['pl_Trock'] = df['pl_Tsubst']*(2/3)**(1/4)

df['ed_1.4_Trock'] = Planck_ratio(df['pl_rprs2'], df['st_teff'], df['pl_Trock'], 1.4e-6)
df['ed_7.5_Trock'] = Planck_ratio(df['pl_rprs2'], df['st_teff'], df['pl_Trock'], 7.5e-6)
df['ed_1.4_Teq'] = Planck_ratio(df['pl_rprs2'], df['st_teff'], df['pl_Teq'], 1.4e-6)
df['ed_7.5_Teq'] = Planck_ratio(df['pl_rprs2'], df['st_teff'], df['pl_Teq'], 7.5e-6)

df['ed_ESM'] = Planck_ratio(df['pl_rprs2'], df['st_teff'], 1.1*df['pl_Teq'], 7.5e-6)

df['ESM'] = 4.29 * df['ed_ESM'] * 10 ** (-0.2*df['sy_kmag'])


# Calculate Transit Depth of Roche Radius

df['q'] = df['pl_bmasse']/df['st_mass'] * MeMs #Mp/M*
df['RR'] = 0.49 * df['q']

def roche_radius(q, ars,rprs):
    return 0.49 * q ** (2 / 3) / (0.6 * q ** (2 / 3) + np.log(1  + q ** (1 / 3))) *ars/rprs

df['RR_RP'] = roche_radius(df['q'], df['pl_ratdor'], df['pl_ratror'])
df['RR_RE'] = df['RR_RP']*df['pl_rade']
df['delta_Na'] = (df['RR_RE']/df['st_rad']*ReRs)**2*1e6






#ind1 = (df['pl_rade'] > 40)
#remove planets with large errors on the mass
ind2 = np.sqrt(df['pl_bmasseerr1']**2+df['pl_bmasseerr2']**2)/df['pl_bmasse'] > 0.5
#remove planets with large errors on the radius
ind3 = np.sqrt(df['pl_radeerr1']**2+df['pl_radeerr2']**2)/df['pl_rade'] > 0.5
#ind4 = df['pl_orbper'] > 10
#ind5 = df['efficiency_kmag'] < 0.3
#ind6 = df['pl_Teq'] < 1000
#ind7 = np.bitwise_and(1500 < df['pl_Teq'], df['pl_Teq'] < 2000)

print([sum(indi) for indi in [ind2,ind3]])
ind = ind2&ind3
df = df[~ind]




# Make trimmed version of table
df_short = df[['pl_name', 'pl_orbper', 'pl_rade', 'pl_ratror', 'pl_rprs2', 'pl_ratdor'
    , 'sy_kmag', 'pl_Teq','pl_Trock',
               'ed_1.4_Trock', 'ed_1.4_Teq',
               'delta_Na', 'efficiency_kmag', 'noise','ESM', 'TSM','disc_facility']]

df_short = df_short.round({'ed_1.4_Trock':1, 'ed_7.5_Trock':1, 'ed_1.4_Teq':1, 'ed_7.5_Teq':1})
df_short = df_short.round({'pl_Teq':1,'pl_Tsubst':1,'pl_Trock':1})


df.to_csv("NASAExoArchive_" + datetime.today().strftime("%Y-%m-%d") + "_aggregate.csv")
df_short.to_csv("NASAExoArchive_" + datetime.today().strftime("%Y-%m-%d") + "_aggregate_short.csv")


with open('table.txt', 'w') as f:
    print(df_short.sort_values(by=['ed_1.4_Trock'], ascending=False).to_string(), file=f)

df_short.sort_values(by=['TSM'], ascending=False).to_csv("test.csv", sep=',')