"""
Feature engineering: 59 physics-based + domain-specific features.

The design is deliberately physics-first: rather than letting the model
re-learn spectral signatures from 1,821 rows, we hand it established remote
sensing indices whose mathematical form encodes decades of domain knowledge.
"""
import numpy as np
import pandas as pd

from .config import BANDS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 59-feature matrix from a (masked) raw frame."""
    out  = pd.DataFrame(index=df.index)
    midx = np.arange(1, 13)
    nd, md, wd, vd, awei_d, ndci_d = {}, {}, {}, {}, {}, {}

    for m in range(1, 13):
        mm   = f'{m:02d}'
        nir  = df[f'nir_{mm}'].values;   red   = df[f'red_{mm}'].values
        green= df[f'green_{mm}'].values;  swir1 = df[f'swir1_{mm}'].values
        swir2= df[f'swir2_{mm}'].values;  nira  = df[f'nira_{mm}'].values
        re1  = df[f're1_{mm}'].values;    blue  = df[f'blue_{mm}'].values
        vv   = df[f'VV_{mm}'].values;     vh    = df[f'VH_{mm}'].values

        nd[mm]    = (nir - red)   / (nir + red   + 1e-10)
        md[mm]    = (green-swir1) / (green+swir1 + 1e-10)
        wd[mm]    = (green-nira)  / (green+nira  + 1e-10)
        vd[mm]    = vh - vv
        awei_d[mm]= blue + 2.5*green - 1.5*(swir1+swir2) - 0.25*nir  # AWEI_sh (Feyisa et al. 2014)
        ndci_d[mm]= (re1 - red) / (re1 + red + 1e-10)

    ndvi  = pd.DataFrame(nd);    mndwi = pd.DataFrame(md)
    ndwi  = pd.DataFrame(wd);    vhvv  = pd.DataFrame(vd)
    awei  = pd.DataFrame(awei_d); ndci  = pd.DataFrame(ndci_d)

    def slope(row):
        mask = row.notna().values
        if mask.sum() < 2: return np.nan
        return np.polyfit(midx[mask], row.values[mask], 1)[0]

    for nm, d in [('NDVI', ndvi), ('MNDWI', mndwi), ('NDWI', ndwi),
                  ('VHVV', vhvv), ('AWEI', awei), ('NDCI', ndci)]:
        out[f'{nm}_mean']  = d.mean(axis=1, skipna=True)
        out[f'{nm}_med']   = d.median(axis=1, skipna=True)
        out[f'{nm}_std']   = d.std(axis=1, skipna=True)
        out[f'{nm}_trend'] = d.apply(slope, axis=1)

    wf  = (mndwi > 0).astype(float).mask(mndwi.isna())
    out['water_persistence']    = wf.mean(axis=1, skipna=True).astype(float)
    nl  = (ndvi < 0.1).astype(float).mask(ndvi.isna())
    out['nonveg_persistence']   = nl.mean(axis=1, skipna=True).astype(float)
    msp = (mndwi > 0.1).astype(float).mask(mndwi.isna())
    out['strong_water_persist'] = msp.mean(axis=1, skipna=True).astype(float)
    awf = (awei > 0).astype(float).mask(awei.isna())
    out['awei_water_persist']   = awf.mean(axis=1, skipna=True).astype(float)

    vh_a = df[[f'VH_{m:02d}' for m in range(1, 13)]]
    vv_a = df[[f'VV_{m:02d}' for m in range(1, 13)]]
    out['vh_mean'] = vh_a.mean(axis=1, skipna=True)
    out['vv_mean'] = vv_a.mean(axis=1, skipna=True)
    out['vh_min']  = vh_a.min(axis=1, skipna=True)
    out['vv_min']  = vv_a.min(axis=1, skipna=True)
    out['vh_std']  = vh_a.std(axis=1, skipna=True)

    cm  = mndwi.notna() & vhvv.notna()
    mc  = mndwi.where(cm); vc = vhvv.where(cm)
    mcc = mc.sub(mc.mean(axis=1), axis=0)
    vcc = vc.sub(vc.mean(axis=1), axis=0)
    num = (mcc * vcc).sum(axis=1)
    den = np.sqrt((mcc**2).sum(axis=1) * (vcc**2).sum(axis=1)) + 1e-10
    out['mndwi_vhvv_corr'] = (num / den).fillna(0.0)

    for nm, d in [('MNDWI', mndwi), ('NDVI', ndvi), ('VHVV', vhvv),
                  ('AWEI', awei), ('NDCI', ndci)]:
        def fv(row): v = row.dropna().values; return v[0]  if len(v) > 0 else np.nan
        def lv(row): v = row.dropna().values; return v[-1] if len(v) > 0 else np.nan
        out[f'{nm}_first']  = d.apply(fv, axis=1)
        out[f'{nm}_last']   = d.apply(lv, axis=1)
        out[f'{nm}_change'] = out[f'{nm}_last'] - out[f'{nm}_first']

    ap = pd.DataFrame(
        {f'm{m}': df[[f'{b}_{m:02d}' for b in BANDS]].notna().any(axis=1).astype(float)
         for m in range(1, 13)})
    out['n_valid_months'] = ap.sum(axis=1)
    def first_m(row):
        for m in range(12):
            if row.iloc[m] == 1: return m + 1
        return np.nan
    out['window_start']  = ap.apply(first_m, axis=1)
    out['window_center'] = out['window_start'] + out['n_valid_months'] / 2.0
    out['start_sin'] = np.sin(2 * np.pi * (out['window_start'] - 1) / 12)
    out['start_cos'] = np.cos(2 * np.pi * (out['window_start'] - 1) / 12)
    out['has_monsoon'] = ap[[f'm{m}' for m in [6, 7, 8, 9]]].any(axis=1).astype(float)
    out['has_dry']     = ap[[f'm{m}' for m in [11, 12, 1, 2, 3]]].any(axis=1).astype(float)

    sar_water = vh_a.apply(lambda col: (col < -26).astype(float)).where(vh_a.notna())
    out['sar_water_persist']   = sar_water.mean(axis=1, skipna=True).fillna(0.0)
    out['sar_optical_diverge'] = out['sar_water_persist'] - out['water_persistence']
    out['mndwi_ndci_interact'] = out['MNDWI_mean'] * out['NDCI_mean']

    return out.replace([np.inf, -np.inf], np.nan)
