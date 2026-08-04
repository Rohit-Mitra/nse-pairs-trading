from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def eg_test(y, x):
    try: _, p, _ = coint(y, x); return p
    except: return 1.0

def joh_test(y, x):
    try:
        r = coint_johansen(np.column_stack([y, x]), 0, 1)
        return r.lr1[0] > r.cvt[0,1] or r.lr2[0] > r.cvm[0,1]
    except: return False

def adf_test(s):
    try: return adfuller(s.dropna(), autolag='AIC')[1]
    except: return 1.0

def half_life(s):
    v = s.dropna().values
    if len(v) < 30: return np.inf
    y, x = np.diff(v), np.column_stack([np.ones(len(v)-1), v[:-1]])
    try:
        lam = np.linalg.lstsq(x, y, rcond=None)[0][1]
        return -np.log(2)/lam if lam < 0 else np.inf
    except: return np.inf

def rs_exponent(s, max_lag=100):
    sc = s.diff().dropna().values
    if len(sc) < max_lag + 10: return 0.5
    ml = min(max_lag, len(sc)//4)
    if ml <= 10: return 0.5
    tau, rs = [], []
    for lag in range(10, ml):
        nc = len(sc)//lag
        if nc < 2: continue
        rsc = []
        for i in range(nc):
            c = sc[i*lag:(i+1)*lag]
            if len(c) < 2: continue
            cd = np.cumsum(c - np.mean(c))
            S = np.std(c, ddof=1)
            if S > 1e-10: rsc.append((np.max(cd)-np.min(cd))/S)
        if len(rsc) >= 2: tau.append(lag); rs.append(np.mean(rsc))
    if len(tau) < 3: return 0.5
    try:
        lt, lr = np.log(np.array(tau, dtype=float)), np.log(np.array(rs, dtype=float))
        v = np.isfinite(lt) & np.isfinite(lr)
        return np.clip(np.polyfit(lt[v], lr[v], 1)[0], 0.01, 0.99) if v.sum()>=3 else 0.5
    except: return 0.5

def ols_hedge(y, x):
    m = OLS(y, add_constant(x)).fit()
    p = m.params
    b = p.iloc[1] if hasattr(p,'iloc') else p[1]
    a = p.iloc[0] if hasattr(p,'iloc') else p[0]
    return b, a

def compute_rsi(s, p=14):
    d = s.diff()
    g, l = d.clip(lower=0).rolling(p).mean(), (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100/(1 + g/l.replace(0, np.nan))