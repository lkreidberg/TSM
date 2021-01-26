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

# ## Read in datasets

#default parameter set = 0
df0 = pd.read_csv(
    "PS_2021.01.26_01.52.28_default_params0.csv",   
    header=128
)

#default parameter set = 1
df1 = pd.read_csv(
    "PS_2021.01.26_01.54.17_default_params1.csv",
    header = 128
)

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

#remove planets with large errors on the mass
ind = df['pl_bmasseerr1']/df['pl_bmasse'] > 0.5
df = df[~ind]

#initialize new column for aggregate temperature
nplanets = len(df)

#first populate with insolation
df['pl_T'] = 255.*(df['pl_insol'])**0.25
#if insolation is unavailable, try equilibrium temperature  
df.pl_T.fillna(df.pl_eqt, inplace=True)
#if equilibrium temperature is unavailable, calculate from a/Rs and Teff
df.pl_T.fillna(1/(np.sqrt(2*df['pl_ratdor']))*df['st_teff'], inplace=True)
#if a/Rs is unavailable, calculate it from a [AU] and Rs [Rsun]
df.pl_T.fillna(1/(np.sqrt(2*215.*df['pl_orbsmax']/df['st_rad']))*df['st_teff'], inplace=True)

#initialize new column for (Rp/Rs)**2
df['pl_rprs2'] = df['pl_ratror']**2
#if rp/rs is unavailable, calculate directly from Rp and Rs
df.pl_rprs2.fillna((ReRs*df['pl_rade']/df['st_rad'])**2, inplace=True)

#scale factor for TSM calculation
df['scale'] = 0
df.loc[df['pl_rade'] <= 1.5, 'scale'] = 0.19
df.loc[df['pl_rade'] > 4, 'scale'] = 1.15 
df.loc[(df['pl_rade'] <= 4)&(df['pl_rade'] >2.75), 'scale'] = 1.28
df.loc[(df['pl_rade'] <= 2.75)&(df['pl_rade'] > 1.5), 'scale'] = 1.26

#initialize new column for TSM
df['TSM'] = df['pl_rade']*df['pl_rprs2']/(ReRs**2)*df['pl_T']/df['pl_bmasse']*10.**(-0.2*df['sy_jmag'])*df['scale']

df['efficiency'] = 1
df.loc[df['sy_jmag'] <= 8.8, 'efficiency'] = (1.375*df['sy_jmag']**2 - 1.214*df['sy_jmag'] - 26.68)/64.58
df.loc[df['sy_jmag'] <= 5., 'efficiency'] = 0.3

#correct TSM for observatinoal efficiency with HST/WFC3
#df['TSM'] = df['TSM']*np.sqrt(df['efficiency'])

df.to_csv("NASAExoArchive_" + datetime.today().strftime("%Y-%m-%d") + "_aggregate.csv")

