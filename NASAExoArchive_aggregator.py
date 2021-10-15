#!/usr/bin/env python
# coding: utf-8

# # TSM Calculator
# 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from astropy import constants as const

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

# ## Build up a filter to remove results from the Stassun et al. 2017 paper, which are often more recent but less precise than previous publications
df.loc[df['pl_refname'].str.contains("STASSUN"), 'pl_refname'].unique()

data_filter = (
    (df['pl_refname'] != '<a refstr=STASSUN_ET_AL__2017 href=https://ui.adsabs.harvard.edu/abs/2017AJ....153..136S/abstract target=ref>Stassun et al. 2017</a>')
)

# ## Group by planet name and grab the most recent record

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

#calculates observational efficiency for HST (accounting for brightness of host star)
df['efficiency'] = 1
df.loc[df['sy_jmag'] <= 8.8, 'efficiency'] = (1.375*df['sy_jmag']**2 - 1.214*df['sy_jmag'] - 26.68)/64.58
df.loc[df['sy_jmag'] <= 5., 'efficiency'] = 0.3

df['efficiency_kmag'] = 1
df.loc[df['sy_kmag'] <= 8.8, 'efficiency_kmag'] = (1.375*df['sy_kmag']**2 - 1.214*df['sy_kmag'] - 26.68)/64.58
df.loc[df['sy_kmag'] <= 5., 'efficiency_kmag'] = 0.3

#option to correct TSM for observatinoal efficiency with HST/WFC3
#df['TSM'] = df['TSM']*np.sqrt(df['efficiency'])

# Fill ars if missing: a(AU)/Rs(Ro)*215
df.pl_ratdor.fillna(df['pl_orbsmax']/df['st_rad']*215, inplace=True)

# Fill insolation if missing: Ts^4/ars^2 * (215^2/5772^4) = Ts^4/ars^2 * 4.166e-11
df.pl_insol.fillna(df['st_teff']**4/df['pl_ratdor']**2*4.166e-11, inplace=True)

#calculates expectecd eclipse depth for the ESM
df['ed_ESM'] = Planck_ratio(df['pl_rprs2'], df['st_teff'], 1.1*df['pl_Teq'], 7.5e-6)

df['ESM'] = 4.29 * df['ed_ESM'] * 10 ** (-0.2*df['sy_kmag'])


###############################################################
# save data frame

df.to_csv("NASAExoArchive_" + datetime.today().strftime("%Y-%m-%d") + "_aggregate.csv")

