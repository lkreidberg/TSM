#!/usr/bin/env python
# coding: utf-8

import numpy as np
# NumPy 2.4+ removed np.in1d; astropy (older) may still reference it.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

import pandas as pd
from datetime import datetime
from astropy import constants as const


def Planck_ratio(depth, Ts, Tp, lam):
    h = const.h.value
    c = const.c.value
    kB = const.k_B.value
    return depth * 1e6 * (np.exp(h * c / (lam * kB * Ts)) - 1) / (np.exp(h * c / (lam * kB * Tp)) - 1)


# Unit conversions
Rjup = const.R_jup / const.R_earth
Mjup = const.M_jup / const.M_earth
ReRs = const.R_earth / const.R_sun
MeMs = const.M_earth / const.M_sun


# ----------------------------
# Read in datasets
# ----------------------------
df0_filename = "default_params0.csv"
df1_filename = "default_params1.csv"

df0 = pd.read_csv(df0_filename, header=0).iloc[:, 1:]
df1 = pd.read_csv(df1_filename, header=0).iloc[:, 1:]

# Concatenate (ignore_index helps keep index clean)
df = pd.concat([df0, df1], ignore_index=True)


# ----------------------------
# Robust pubdate parsing
# ----------------------------
# Original code tried multiple formats with errors="ignore" then errors="raise".
# With mixed inputs, pandas can crash internally (AssertionError).
#
# Approach:
# 1) extract a YYYY-MM, YYYY-MM-DD, or YYYY-MM-DD HH:MM substring if present
# 2) parse with to_datetime(errors="coerce") so bad rows become NaT
#
# This preserves sorting for "most recent record" logic without crashing.

pub = df["pl_pubdate"].astype("string")

# Prefer full timestamp if present, else date, else year-month
ts = pub.str.extract(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})", expand=False)
d  = pub.str.extract(r"(\d{4}-\d{2}-\d{2})", expand=False)
ym = pub.str.extract(r"(\d{4}-\d{2})", expand=False)

parsed = pd.to_datetime(ts, errors="coerce")
parsed = parsed.fillna(pd.to_datetime(d, errors="coerce"))
parsed = parsed.fillna(pd.to_datetime(ym, format="%Y-%m", errors="coerce"))

df = df.assign(dt_obj=parsed)

# Sort newest first (NaT will sort last by default)
df = df.sort_values(by="dt_obj", ascending=False)


# ----------------------------
# Filter out Stassun et al. 2017 (less precise)
# ----------------------------
stassun_ref = (
    '<a refstr=STASSUN_ET_AL__2017 '
    'href=https://ui.adsabs.harvard.edu/abs/2017AJ....153..136S/abstract '
    'target=ref>Stassun et al. 2017</a>'
)

# Keep rows that are NOT that ref (also handles NaN safely)
data_filter = df["pl_refname"].fillna("") != stassun_ref


# ----------------------------
# Group by planet name and grab the most recent record
# ----------------------------
# After sorting by dt_obj desc, 'first' picks the most recent entry per planet.
cols = df.columns.to_list()[1:]  # keep your original behavior: skip first column
agg_dict = dict.fromkeys(cols, "first")

df = (
    df.loc[data_filter]
      .groupby("pl_name", as_index=False)
      .agg(agg_dict)
)

# Optional: defragment (can help performance after heavy ops)
df = df.copy()


# ----------------------------
# Fill derived quantities
# ----------------------------
# Fill in columns where mass or radius are only in Jupiter units
df["pl_rade"] = df["pl_rade"].fillna(df["pl_radj"] * Rjup)
df["pl_bmasse"] = df["pl_bmasse"].fillna(df["pl_bmassj"] * Mjup)

# Planet temperature aggregate
df["pl_Teq"] = 278.0 * (df["pl_insol"]) ** 0.25
df["pl_Teq"] = df["pl_Teq"].fillna(df["pl_eqt"])
df["pl_Teq"] = df["pl_Teq"].fillna(1 / (np.sqrt(2 * df["pl_ratdor"])) * df["st_teff"])
df["pl_Teq"] = df["pl_Teq"].fillna(1 / (np.sqrt(2 * 215.0 * df["pl_orbsmax"] / df["st_rad"])) * df["st_teff"])

# Fill rprs if not given
df["pl_ratror"] = ReRs * df["pl_rade"] / df["st_rad"]
df["pl_rprs2"] = df["pl_ratror"] ** 2

# Scale factor for TSM calculation (Kempton+ 2018)
df["scale"] = 0.0
df.loc[df["pl_rade"] <= 1.5, "scale"] = 0.19
df.loc[df["pl_rade"] > 4.0, "scale"] = 1.15
df.loc[(df["pl_rade"] <= 4.0) & (df["pl_rade"] > 2.75), "scale"] = 1.28
df.loc[(df["pl_rade"] <= 2.75) & (df["pl_rade"] > 1.5), "scale"] = 1.26

# TSM
df["TSM"] = (
    df["pl_rade"] * df["pl_rprs2"] / (ReRs ** 2)
    * df["pl_Teq"] / df["pl_bmasse"]
    * 10.0 ** (-0.2 * df["sy_jmag"])
    * df["scale"]
)

# Observational efficiency (HST brightness correction)
df["efficiency"] = 1.0
df.loc[df["sy_jmag"] <= 8.8, "efficiency"] = (1.375 * df["sy_jmag"] ** 2 - 1.214 * df["sy_jmag"] - 26.68) / 64.58
df.loc[df["sy_jmag"] <= 5.0, "efficiency"] = 0.3

df["efficiency_kmag"] = 1.0
df.loc[df["sy_kmag"] <= 8.8, "efficiency_kmag"] = (1.375 * df["sy_kmag"] ** 2 - 1.214 * df["sy_kmag"] - 26.68) / 64.58
df.loc[df["sy_kmag"] <= 5.0, "efficiency_kmag"] = 0.3

# Option to correct TSM for observational efficiency with HST/WFC3
# df["TSM"] = df["TSM"] * np.sqrt(df["efficiency"])

# Fill ars if missing: a(AU)/Rs(Ro)*215
df["pl_ratdor"] = df["pl_ratdor"].fillna(df["pl_orbsmax"] / df["st_rad"] * 215.0)

# Fill insolation if missing:
# Ts^4/ars^2 * (215^2/5772^4) = Ts^4/ars^2 * 4.166e-11
df["pl_insol"] = df["pl_insol"].fillna(df["st_teff"] ** 4 / df["pl_ratdor"] ** 2 * 4.166e-11)

m = 1.33
b = 1660
Tbar = 1524
df["pl_tday"] = m * (df["pl_Teq"] - Tbar) + b

# Eclipse depths / ESM
df["ed_ESM"] = Planck_ratio(df["pl_rprs2"], df["st_teff"], 1.1 * df["pl_Teq"], 7.5e-6)
df["ed_ESM_night"] = Planck_ratio(df["pl_rprs2"], df["st_teff"], 1000.0, 7.5e-6)
df["ed_ESM_day"] = Planck_ratio(df["pl_rprs2"], df["st_teff"], df["pl_tday"], 7.5e-6)

df["ESM"] = 4.29 * df["ed_ESM"] * 10.0 ** (-0.2 * df["sy_kmag"])
df["SN_night"] = df["ed_ESM_night"] * 10.0 ** (-0.2 * df["sy_kmag"])
df["SN_day"] = df["ed_ESM_day"] * 10.0 ** (-0.2 * df["sy_kmag"])


# ----------------------------
# Save
# ----------------------------
outname = "NASAExoArchive_" + datetime.today().strftime("%Y-%m-%d") + "_aggregate.csv"
df.to_csv(outname, index=False)
print(f"Wrote: {outname}")
