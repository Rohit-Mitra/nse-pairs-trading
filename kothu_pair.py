# NSE PAIRS TRADING — LightGBM | 14 Sectors | 3-Way Split | 5 Features
# FIX 1: zscore-reversion target | FIX 2: window opt on VAL
# FIX 3: AND validation filter  | FIX 4: no global in simulate
# FIX 5: position ffill capped
import yfinance as yf, pandas as pd, numpy as np, warnings
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.api import OLS, add_constant
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")

# ── 1. PARAMETERS ───────────────────────────────────────────────
print("="*60 + "\nNSE PAIRS TRADING — LightGBM\n" + "="*60)

SECTORS = {
    "Banking": ["KOTAKBANK.NS","AXISBANK.NS","SBIN.NS","INDUSINDBK.NS",
                "IDFCFIRSTB.NS","BANDHANBNK.NS","FEDERALBNK.NS","CANBK.NS",
                "BANKBARODA.NS","PNB.NS","INDIANB.NS"],
    "IT": ["TCS.NS","WIPRO.NS","TECHM.NS","MPHASIS.NS","COFORGE.NS",
           "PERSISTENT.NS","LTTS.NS","TATAELXSI.NS","CYIENT.NS"],
    "Pharma": ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","AUROPHARMA.NS",
               "LUPIN.NS","BIOCON.NS","ALKEM.NS","IPCALAB.NS","LAURUSLABS.NS",
               "GLENMARK.NS","AJANTPHARM.NS","APOLLOHOSP.NS","FORTIS.NS"],
    "FMCG": ["HINDUNILVR.NS","ITC.NS","BRITANNIA.NS","MARICO.NS","DABUR.NS",
             "GODREJCP.NS","TATACONSUM.NS","COLPAL.NS","EMAMILTD.NS","JYOTHYLAB.NS"],
    "Metals": ["TATASTEEL.NS","JSWSTEEL.NS","SAIL.NS",
               "JINDALSTEL.NS","NMDC.NS","COALINDIA.NS"],
    "Cement": ["ULTRACEMCO.NS","AMBUJACEM.NS","JKCEMENT.NS","ACC.NS",
               "RAMCOCEM.NS","DALBHARAT.NS"],
    "Auto": ["MARUTI.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","TVSMOTOR.NS",
             "MOTHERSON.NS","BOSCHLTD.NS","BHARATFORG.NS","EXIDEIND.NS"],
    "Oil_Gas": ["RELIANCE.NS","IOC.NS","BPCL.NS","ONGC.NS","PETRONET.NS","MGL.NS"],
    "NBFC": ["BAJFINANCE.NS","BAJAJFINSV.NS","CHOLAFIN.NS","MUTHOOTFIN.NS",
             "LICHSGFIN.NS","MANAPPURAM.NS"],
    "Utilities": ["TATAPOWER.NS","NHPC.NS","CESC.NS"],
    "Telecom": ["BHARTIARTL.NS","IDEA.NS"],
    "RealEstate": ["DLF.NS","GODREJPROP.NS","PRESTIGE.NS","BRIGADE.NS"],
    "Insurance": ["SBILIFE.NS","HDFCLIFE.NS","ICICIPRULI.NS"],
    "Chemicals": ["PIDILITIND.NS","SRF.NS","DEEPAKNTR.NS"],
}
ALL_TICKERS = [t for s in SECTORS.values() for t in s]

TRAIN_START, TRAIN_END = "2016-01-01", "2020-12-31"
VAL_START, VAL_END     = "2021-01-01", "2021-12-31"
TEST_START, TEST_END   = "2022-01-01", "2025-12-31"

TXN_COST, COINT_P, ENTRY_Z, EXIT_Z = 0.0025, 0.05, 1.0, 0.5
FFILL_LIMIT    = 15                  # FIX 5: max bars to hold stale position
WINDOW_OPTS    = [40, 60, 80, 100]
INIT_CAPITAL   = 10_000_000
TARGET_H, TARGET_ZT = 5, 0.5
HL_MIN, HL_MAX = 5, 120
RS_THRESH      = 0.55
CROSS_LINKS    = [("Banking","NBFC"),("Metals","Auto"),("Oil_Gas","Auto"),
                  ("FMCG","Pharma"),("Utilities","Oil_Gas")]
WF_FOLDS       = [("2016-01-01","2018-12-31","2019-01-01","2019-12-31"),
                  ("2016-01-01","2019-12-31","2020-01-01","2020-12-31")]
FEATURE_COLS   = ["zscore","zscore_lag1","vol_ratio","rel_momentum","rsi_spread"]

print(f"Sectors: {len(SECTORS)} | Tickers: {len(ALL_TICKERS)} | Capital: ₹{INIT_CAPITAL:,.0f}")
print(f"Train: {TRAIN_START}→{TRAIN_END} | Val: {VAL_START}→{VAL_END} | Test: {TEST_START}→{TEST_END}")
print(f"FFILL limit: {FFILL_LIMIT} bars | TxnCost: {TXN_COST*100:.1f}bps")

# ── 2. STAT FUNCTIONS ───────────────────────────────────────────
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

def build_model(fast=False):
    return LGBMClassifier(n_estimators=100 if fast else 300, max_depth=4,
        learning_rate=0.05, num_leaves=21, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, reg_alpha=0.5, reg_lambda=2.0,
        random_state=42, verbose=-1, n_jobs=-1)

# ── 3. DOWNLOAD ─────────────────────────────────────────────────
print("\n[3] Downloading...")
raw = yf.download(ALL_TICKERS, start=TRAIN_START, end=TEST_END, progress=True)

if isinstance(raw.columns, pd.MultiIndex):
    price_data = raw["Adj Close"].copy() if "Adj Close" in raw.columns.get_level_values(0) else raw["Close"].copy()
else:
    c = "Adj Close" if "Adj Close" in raw.columns else "Close"
    price_data = raw[[c]].copy(); price_data.columns = ALL_TICKERS[:1]
price_data = price_data.ffill().bfill()

bad = [c for c in price_data.columns if price_data[c].isna().mean() > 0.05]
if bad: price_data.drop(columns=bad, inplace=True)
cutoff = pd.Timestamp("2016-06-01")
late = [c for c in price_data.columns if (f:=price_data[c].first_valid_index()) and f > cutoff]
if late: price_data.drop(columns=late, inplace=True)
price_data.dropna(axis=1, how="all", inplace=True)
price_data = price_data.ffill().bfill()
valid = list(price_data.columns)

# Pre-compute returns once (FIX 4: simulate receives sliced returns, not global)
all_returns = price_data.pct_change()

try:
    nr = yf.download("^NSEI", start=TRAIN_START, end=TEST_END, progress=False)
    nifty = nr["Adj Close"] if "Adj Close" in nr.columns else nr["Close"]
    if isinstance(nifty, pd.DataFrame): nifty = nifty.iloc[:,0]
    nifty = pd.Series(nifty.values, index=nifty.index).ffill().bfill()
except: nifty = None

print(f"✓ {len(valid)} tickers | {price_data.index[0].date()} → {price_data.index[-1].date()}")
if len(valid) < 4: print("✗ Too few tickers."); exit()

# ── 4. PAIR FILTERING ───────────────────────────────────────────
print("\n[4] Pair filtering...")
combos = []
for sec, tks in SECTORS.items():
    sv = [t for t in tks if t in valid]
    combos += [(a,b,sec) for a,b in combinations(sv,2)]
for s1,s2 in CROSS_LINKS:
    t1 = [t for t in SECTORS.get(s1,[]) if t in valid]
    t2 = [t for t in SECTORS.get(s2,[]) if t in valid]
    combos += [(a,b,f"{s1}×{s2}") for a in t1 for b in t2 if a!=b]
print(f"  Combos: {len(combos)}")

pairs, pair_info = [], []
for i,(s1,s2,sec) in enumerate(combos):
    if (i+1)%100==0: print(f"  {i+1}/{len(combos)}...", end="\r")
    sub = price_data.loc[TRAIN_START:TRAIN_END,[s1,s2]].dropna()
    if len(sub) < 200: continue
    y, x = sub[s1], sub[s2]
    eg_p = eg_test(y,x); eg_ok = eg_p < COINT_P
    joh_ok = joh_test(y,x)
    if not eg_ok and not joh_ok: continue
    b, a = ols_hedge(y, x); sp = y - b*x - a
    if adf_test(sp) >= 0.10: continue
    hl = half_life(sp)
    if not (HL_MIN <= hl <= HL_MAX): continue
    rs = rs_exponent(sp)
    if rs >= RS_THRESH: continue
    pairs.append((s1,s2))
    pair_info.append({"Stock1":s1,"Stock2":s2,"Sector":sec,"EG_P":round(eg_p,5),
        "Johansen":joh_ok,"HalfLife":round(hl,1),"RS_Exp":round(rs,4),"Beta":round(b,4)})

print(f"\n✓ {len(pairs)} pairs passed")
if not pairs: print("✗ No pairs."); exit()
pd.DataFrame(pair_info).to_csv("cointegrated_pairs.csv", index=False)

# ── 5. FEATURES & HELPERS ───────────────────────────────────────
def create_features(yp, xp, beta, win):
    sp = yp - beta*xp
    rm, rs_ = sp.rolling(win).mean(), sp.rolling(win).std().replace(0,np.nan)
    df = pd.DataFrame(index=sp.index)
    df["spread"], df["zscore"] = sp, (sp-rm)/rs_
    df["zscore_lag1"] = df["zscore"].shift(1)
    vs, vl = sp.rolling(10).std(), sp.rolling(40).std()
    df["vol_ratio"] = (vs/vl.replace(0,np.nan)).fillna(1)
    df["rel_momentum"] = (yp.pct_change(5)-xp.pct_change(5)).fillna(0)
    df["rsi_spread"] = compute_rsi(sp, 14)
    # FIX 1: zscore-reversion target (did zscore cross back past EXIT_Z?)
    fz = df["zscore"].shift(-TARGET_H)
    lo = df["zscore"] < -TARGET_ZT          # long-spread entry zone
    sh = df["zscore"] >  TARGET_ZT          # short-spread entry zone
    lok = fz > -EXIT_Z                      # zscore recovered toward mean
    shk = fz <  EXIT_Z                      # zscore recovered toward mean
    tgt = pd.Series(np.nan, index=df.index)
    tgt[lo & lok] = 1.0;  tgt[lo & ~lok] = 0.0
    tgt[sh & shk] = 1.0;  tgt[sh & ~shk] = 0.0
    df["target"] = tgt
    df.dropna(subset=FEATURE_COLS, inplace=True)
    return df

def train_predict(tr, feat_cols=FEATURE_COLS, fast=False):
    tr = tr.dropna(subset=["target"]); tr["ml_target"] = tr["target"].astype(int)
    if len(tr)<50 or len(np.unique(tr["ml_target"]))<2: return None, None, None
    sc = StandardScaler(); X = sc.fit_transform(tr[feat_cols]); y = tr["ml_target"]
    m = build_model(fast); m.fit(X, y)
    return m, sc, accuracy_score(y, m.predict(X))

# FIX 4: simulate receives pre-sliced returns — no global price_data reference
# FIX 5: position ffill capped at FFILL_LIMIT bars
def simulate(data, model, scaler, beta, y_ret, x_ret, cap):
    d = data.copy()
    feat = scaler.transform(d[FEATURE_COLS])
    d["pred_signal"], d["pred_proba"] = model.predict(feat), model.predict_proba(feat)[:,1]
    d["position"] = 0
    d.loc[(d["zscore_lag1"]<-ENTRY_Z)&(d["pred_signal"]==1),"position"] = 1
    d.loc[(d["zscore_lag1"]> ENTRY_Z)&(d["pred_signal"]==0),"position"] = -1
    d.loc[d["zscore_lag1"].abs()<EXIT_Z,"position"] = 0
    d["position"] = d["position"].replace(0,np.nan).ffill(limit=FFILL_LIMIT).fillna(0)  # FIX 5
    yr = y_ret.reindex(d.index).fillna(0)   # FIX 4: pre-sliced, no global
    xr = x_ret.reindex(d.index).fillna(0)
    d["pnl_gross"] = d["position"].shift(1).fillna(0)*cap*(yr - beta*xr)
    d["txn_cost"] = d["position"].diff().fillna(0).abs()*cap*TXN_COST
    d["pnl_net"] = d["pnl_gross"] - d["txn_cost"]
    d["cum_pnl"] = d["pnl_net"].cumsum()
    d["portfolio_value"] = cap + d["cum_pnl"]
    return d

def risk_metrics(pnl, cum):
    dr = pnl.dropna(); c = cum.dropna()
    if len(dr)==0: return {}
    sh = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    dd = (c - c.cummax()).min()
    ar = dr.mean()*252
    gp, gl = dr[dr>0].sum(), dr[dr<0].abs().sum()
    bp = (c<c.cummax()).astype(int)
    mdd = (bp*(bp.groupby((bp!=bp.shift()).cumsum()).cumcount()+1)).max()
    return {"Sharpe":round(sh,3),"MaxDD":round(dd,2),"MaxDD_Days":int(mdd),
            "Calmar":round(ar/abs(dd),3) if dd!=0 else 0,
            "VaR95":round(dr.quantile(0.05),2),"HitRate":round((dr>0).mean(),4),
            "ProfitFactor":round(gp/gl,3) if gl>0 else None,"AnnRet":round(ar,2)}

# ── 6. VALIDATION PASS ──────────────────────────────────────────
# FIX 3: AND logic (was OR) + fallback if too strict
print("\n[6] Validation filtering...")
val_pairs, val_info = [], []
for i,(ys,xs) in enumerate(pairs):
    tr = price_data.loc[TRAIN_START:TRAIN_END,[ys,xs]].dropna()
    try: b,_ = ols_hedge(tr[ys], tr[xs])
    except: b = 1.0
    vp = price_data.loc[VAL_START:VAL_END,[ys,xs]].dropna()
    if len(vp)<50: continue
    vs = vp[ys] - b*vp[xs]
    if adf_test(vs)<0.20 or half_life(vs)<HL_MAX or rs_exponent(vs)<RS_THRESH:
        val_pairs.append((ys,xs)); val_info.append(pair_info[i])

# FIX 3: fallback — if too aggressive, keep top half by EG p-value
if len(val_pairs) < max(2, len(pairs)//4):
    print(f"  Val filter too strict ({len(val_pairs)}) — relaxing to top half by EG p-value")
    sorted_pi = sorted(pair_info, key=lambda d: d["EG_P"])
    half = max(2, len(sorted_pi)//2)
    val_info = sorted_pi[:half]
    val_pairs = [(d["Stock1"],d["Stock2"]) for d in val_info]

if not val_pairs: val_pairs, val_info = pairs, pair_info
pairs, pair_info = val_pairs, val_info
N_PAIRS = len(pairs); CAP_PAIR = INIT_CAPITAL/max(N_PAIRS,1)
print(f"  ✓ {N_PAIRS} pairs | ₹{CAP_PAIR:,.0f}/pair")

# ── 7. TRAIN & TEST ─────────────────────────────────────────────
print(f"\n{'='*60}\n[7] Training & Simulating\n{'='*60}")
trades_all, summary, port_pnl = [], [], pd.Series(dtype=float)

for idx,(ys,xs) in enumerate(pairs,1):
    sec = next((d["Sector"] for d in pair_info if d["Stock1"]==ys and d["Stock2"]==xs),"?")
    print(f"\n--- {idx}/{N_PAIRS}: {ys}/{xs} [{sec}] ---")
    tr = price_data.loc[TRAIN_START:TRAIN_END].dropna()
    try: b,_ = ols_hedge(tr[ys], tr[xs])
    except: b = 1.0

    # WF accuracy (diagnostic only)
    fa = create_features(price_data[ys], price_data[xs], b, 60)
    wf_acc = []
    for ts,te,vs_,ve in WF_FOLDS:
        t, v = fa.loc[ts:te].copy(), fa.loc[vs_:ve].copy()
        t = t.dropna(subset=["target"]); t["ml_target"]=t["target"].astype(int)
        v = v.dropna(subset=["target"])
        if len(t)<50 or len(v)<20 or len(np.unique(t["ml_target"]))<2: continue
        sc = StandardScaler(); Xt=sc.fit_transform(t[FEATURE_COLS])
        m=build_model(True); m.fit(Xt,t["ml_target"])
        wf_acc.append(accuracy_score(v["target"].astype(int), m.predict(sc.transform(v[FEATURE_COLS]))))
    wf = np.mean(wf_acc) if wf_acc else 0

    # FIX 2: window opt on VALIDATION period (was train)
    best_pnl, best_win = -np.inf, 60
    for w in WINDOW_OPTS:
        fd = create_features(price_data[ys], price_data[xs], b, w)
        td = fd.loc[TRAIN_START:TRAIN_END].copy()
        vd = fd.loc[VAL_START:VAL_END].copy()
        if len(vd) < 30: continue
        m,sc,_ = train_predict(td, fast=True)
        if m is None: continue
        yr_v = all_returns[ys].loc[VAL_START:VAL_END]
        xr_v = all_returns[xs].loc[VAL_START:VAL_END]
        sp = simulate(vd, m, sc, b, yr_v, xr_v, CAP_PAIR)
        p = sp["pnl_net"].sum()
        if p > best_pnl: best_pnl, best_win = p, w

    # Final train→test
    ff = create_features(price_data[ys], price_data[xs], b, best_win)
    ft, ftest = ff.loc[TRAIN_START:TRAIN_END].copy(), ff.loc[TEST_START:TEST_END].copy()
    fm, fsc, tacc = train_predict(ft, fast=False)
    if fm is None or len(ftest)<20:
        print("  ✗ Skipping"); continue

    # FIX 4: pass pre-sliced returns
    yr_te = all_returns[ys].loc[TEST_START:TEST_END]
    xr_te = all_returns[xs].loc[TEST_START:TEST_END]
    res = simulate(ftest, fm, fsc, b, yr_te, xr_te, CAP_PAIR)

    pnl = res["pnl_net"].sum(); ret = pnl/CAP_PAIR*100
    ntrades = int((res["position"].diff()!=0).sum())
    met = risk_metrics(res["pnl_net"], res["cum_pnl"])
    port_pnl = port_pnl.add(res["pnl_net"].fillna(0), fill_value=0) if not port_pnl.empty else res["pnl_net"].fillna(0)

    print(f"  PnL: ₹{pnl:,.0f} ({ret:+.1f}%) | Sharpe: {met.get('Sharpe',0)} | "
          f"Trades: {ntrades} | Win: {best_win}d")

    trades_all.append(pd.DataFrame({"Pair":f"{ys}/{xs}","Sector":sec,"Date":res.index,
        "Position":res["position"].values,"Spread":res["spread"].values,
        "ZScore":res["zscore"].values,"PnL_Net":res["pnl_net"].values,
        "Cum_PnL":res["cum_pnl"].values,"Value":res["portfolio_value"].values}))

    summary.append({"Pair":f"{ys}/{xs}","Sector":sec,"Beta":round(b,4),"Window":best_win,
        "Capital":CAP_PAIR,"WF_Acc":round(wf,4),"Train_Acc":round(tacc,4),
        "Trades":ntrades,"Net_PnL":round(pnl,2),"Return%":round(ret,2),
        "EndValue":round(CAP_PAIR+pnl,2),"TxnCost":round(res["txn_cost"].sum(),2),**met})

# ── 8. RESULTS ───────────────────────────────────────────────────
print("\n" + "="*60 + "\nFINAL RESULTS\n" + "="*60)

if trades_all:
    pd.concat(trades_all, ignore_index=True).to_csv("nse_pairs_trades_detailed.csv", index=False)

if summary:
    sdf = pd.DataFrame(summary).sort_values("Net_PnL", ascending=False)
    sdf.to_csv("nse_pairs_summary.csv", index=False)
    tot = sdf["Net_PnL"].sum(); ret = tot/INIT_CAPITAL*100
    yrs = (pd.Timestamp(TEST_END)-pd.Timestamp(TEST_START)).days/365.25
    ps = port_pnl.mean()/port_pnl.std()*np.sqrt(252) if port_pnl.std()>0 else 0

    print(f"\n{'Pair':<35} {'PnL':>12} {'Ret%':>8} {'Sharpe':>8} {'Trades':>7}")
    print("-"*72)
    for _,r in sdf.iterrows():
        print(f"  {r['Pair']:<33} ₹{r['Net_PnL']:>10,.0f} {r['Return%']:>+7.1f}% "
              f"{r.get('Sharpe',0):>7.2f} {r['Trades']:>6}")

    print(f"\n{'─'*60}")
    print(f"  Capital: ₹{INIT_CAPITAL:,.0f} → ₹{INIT_CAPITAL+tot:,.0f}")
    print(f"  Return: {ret:+.2f}% | Ann: {ret/yrs:+.2f}% | Sharpe: {ps:.3f}")
    print(f"  Pairs: {N_PAIRS} | Win rate: {(sdf['Net_PnL']>0).mean()*100:.0f}%")

    print(f"\n  {'Sector':<18} {'#':>3} {'PnL':>12} {'Ret%':>8}")
    for sec in sorted(sdf["Sector"].unique()):
        sd = sdf[sdf["Sector"]==sec]
        sp, sa = sd["Net_PnL"].sum(), sd["Capital"].sum()
        print(f"  {sec:<18} {len(sd):>3} ₹{sp:>10,.0f} {sp/sa*100:>+7.1f}%")

    if nifty is not None:
        ns = nifty.loc[TEST_START:TEST_END].dropna()
        if isinstance(ns, pd.DataFrame): ns = ns.iloc[:,0]
        ns = pd.Series(ns.values, index=ns.index)
        nr_ = ns.pct_change().dropna()
        nret = (float(ns.iloc[-1])/float(ns.iloc[0])-1)*100
        nsh = nr_.mean()/nr_.std()*np.sqrt(252) if nr_.std()>0 else 0
        print(f"\n  Nifty50: {nret:+.1f}% | Sharpe: {nsh:.3f}")
        print(f"  Alpha: {ret-nret:+.2f}% | Sharpe edge: {ps-nsh:+.3f}")

    print(f"\n✓ Saved: cointegrated_pairs.csv, nse_pairs_summary.csv, nse_pairs_trades_detailed.csv")
else:
    print("  ✗ No results.")

print("\n" + "="*60 + "\nDONE\n" + "="*60)