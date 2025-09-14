#########################################################
### Step 0                                             ##
### Select your assets, data range and risk-free rate  ##
#########################################################

# 1) Installs (in their own cell; no comments on the same line)
# !pip install pandas-datareader
# !pip install scipy
# !pip install --upgrade yfinance

# 2) Imports
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.optimize as sco

# 3) Parameters
symbols = ['SPY','TLT','IEF','GLD']  
# 4 Equity Market Index => ['SPY', 'FXI', 'EWQ', 'EWG']
# All Wether Portfolio  => ['SPY','TLT','IEF','GLD']
# Stock Pick Genius     => ['GLD','SPY','MSFT','AAPL']
start_date = '2005-01-01'
end_date   = '2025-08-31'
risk_free_rate = 0.03   # assumed annual risk-free rate

# 4) Download adjusted close prices
data = yf.download(symbols, start=start_date, end=end_date, auto_adjust=False)['Adj Close']

# --- Ensure column order matches `symbols` (drop tickers not available) ---
if isinstance(data, pd.Series):
    data = data.to_frame()  # in case of single ticker
cols_in_data = [s for s in symbols if s in data.columns]
data = data.loc[:, cols_in_data].dropna(how='all')

# 5) Charts
# (a) Price indexed to 100
(data / data.iloc[0] * 100).plot(figsize=(8,5), title='Price (Indexed to 100)')
plt.ylabel('Index Level')
plt.tight_layout()
plt.show()

# (b) Daily simple return distributions
rets = data.pct_change().dropna()
rets.hist(bins=40, figsize=(8,5))
plt.suptitle('Daily Simple Return Distributions')
plt.tight_layout()
plt.show()

# 6) Annualized statistics (simple returns, population moments)
annual_ret  = (rets.mean() * 252).reindex(data.columns)
annual_std  = (rets.std(ddof=0) * math.sqrt(252)).reindex(data.columns)
annual_cov  = (rets.cov(ddof=0) * 252).reindex(index=data.columns, columns=data.columns)
annual_corr =  rets.corr().reindex(index=data.columns, columns=data.columns)

# 7) Sharpe Ratios (arithmetic definition)
sharpe_ratio = (annual_ret - risk_free_rate) / annual_std

# 8) Summary table
summary_table = pd.DataFrame({
    'R':  annual_ret.round(4),
    'SD': annual_std.round(4),
    'SR': sharpe_ratio.round(4)
})

print("\n=== Annualized Returns, Standard Deviations, and Sharpe Ratios ===")
print(summary_table)

print("\n=== Annualized Covariance Matrix ===")
print(annual_cov.round(4))

print("\n=== Correlation Matrix ===")
print(annual_corr.round(4))


#########################################################
### Step 1-1  Investment Opportunity Set               ##
#########################################################

def port_ret(weights):
    return weights.T @ annual_ret

def port_vol(weights):
    return (weights.T @ annual_cov @ weights) ** 0.5

noa = len(data.columns)

# Monte Carlo
prets, pvols = [], []
for _ in range(2500):
    w = np.random.random(noa)
    w /= w.sum()
    prets.append(port_ret(w))
    pvols.append(port_vol(w))
prets = np.array(prets)
pvols = np.array(pvols)

# Individual assets
asset_ret = annual_ret.values
asset_vol = annual_std.values

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(pvols, prets, c=(prets - risk_free_rate) / pvols,
            marker='o', cmap='coolwarm', alpha=0.85)
plt.scatter(asset_vol, asset_ret, s=70, facecolors='yellow', edgecolors='black', zorder=4)

for x, y, name in zip(asset_vol, asset_ret, annual_ret.index):
    plt.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6),
                 ha='left', fontsize=9, zorder=5)

plt.grid(True, alpha=0.3)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.title('Investment Opportunity Set')
plt.show()

#########################################################
### Step 1-2  Efficient Frontier                       ##
#########################################################

# MVP
bnds = tuple((0, 1) for _ in range(noa))
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
eweights = np.array(noa * [1. / noa,])
optv = sco.minimize(port_vol, eweights, method='SLSQP',
                    bounds=bnds, constraints=cons)

min_return = port_ret(optv['x'])
max_return = np.max(annual_ret)

trets = np.linspace(min_return, max_return, 50)
tvols = []
for tret in trets:
    cons = ({'type': 'eq', 'fun': lambda x: port_ret(x) - tret},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    res = sco.minimize(port_vol, eweights, method='SLSQP',
                       bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)

# MVP stats
mvp_w   = optv.x
mvp_vol = float(port_vol(mvp_w))
mvp_ret = float(port_ret(mvp_w))
mvp_sr  = float((mvp_ret - risk_free_rate) / mvp_vol)

weights_s = pd.Series(mvp_w, index=data.columns)
weights_s[weights_s.abs() < 1e-6] = 0.0
weights_df = pd.DataFrame({"Weight (%)": (weights_s * 100).round(2)}).sort_values("Weight (%)", ascending=False)

# Plot with text
fig = plt.figure(figsize=(7, 8), constrained_layout=True)
gs  = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[6, 1], hspace=0.03)

ax = fig.add_subplot(gs[0])
ax_txt = fig.add_subplot(gs[1])
ax_txt.axis('off')

ax.scatter(pvols, prets, c=(prets - risk_free_rate) / pvols,
           marker='o', alpha=0.85, cmap='coolwarm', label='Random Portfolios')
ax.plot(tvols, trets, 'b', lw=4.0, label='Efficient Frontier')
ax.plot(mvp_vol, mvp_ret, 'ks', markersize=8.0, label="MVP")

ax.scatter(annual_std, annual_ret, s=70, facecolors='yellow',
           edgecolors='black', zorder=4, label='Assets')
for x, y, name in zip(annual_std.values, annual_ret.values, annual_ret.index):
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6), ha='left', fontsize=9, zorder=5)

ax.grid(True, alpha=0.3)
ax.set_xlabel('expected volatility')
ax.set_ylabel('expected return')
ax.set_title("Efficient Frontier")
ax.legend(loc='best')

result_text = (
    "================ Minimum-Variance Portfolio (Long-only) ================\n\n"
    "Assets & Weights (%):\n"
    f"{weights_df.to_string()}\n\n"
    "Summary Stats:\n"
    f"- Volatility (σ):  {mvp_vol:.3f}\n"
    f"- Return (μ):      {mvp_ret:.3f}\n"
    f"- Sharpe Ratio:    {mvp_sr:.3f}\n"
    "=======================================================================\n"
)
ax_txt.text(0.5, 1.0, result_text, ha='center', va='top',
            fontsize=9, family='monospace', transform=ax_txt.transAxes)

plt.show()


#########################################################
### Step 2                                             ##
### Optimal Risk Portfolio and Capital Market Line     ##
#########################################################

import scipy.optimize as sco

# ---------- ORP (Max Sharpe) ----------
def min_func_sharpe(weights):
    return -(port_ret(weights) - risk_free_rate) / port_vol(weights)

noa = len(annual_ret)
bnds = tuple((0, 1) for _ in range(noa))
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
eweights = np.array(noa * [1.0 / noa,])

opts = sco.minimize(min_func_sharpe, eweights, method='SLSQP',
                    bounds=bnds, constraints=cons)

# ORP metrics
orp_w   = opts.x
orp_vol = float(port_vol(orp_w))
orp_ret = float(port_ret(orp_w))
orp_sr  = float((orp_ret - risk_free_rate) / orp_vol)

# ORP weights (show % only)
orp_weights_s = pd.Series(orp_w, index=annual_ret.index)
orp_weights_s[orp_weights_s.abs() < 1e-6] = 0.0
orp_weights_df = pd.DataFrame(
    {"Weight (%)": (orp_weights_s * 100).round(2)}
).sort_values("Weight (%)", ascending=False)

# ---------- Capital Market Line (no fsolve needed) ----------
# CML: y = rf + (Sharpe_orp) * x
cml_x = np.linspace(0.0, float(max(pvols.max(), orp_vol)) * 1.05, 200)
cml_y = risk_free_rate + orp_sr * cml_x

# ---------- Plot: ORP & CML ----------
fig = plt.figure(figsize=(7, 8), constrained_layout=True)
gs  = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[6, 1], hspace=0.03)

ax = fig.add_subplot(gs[0])        # Top: graph
ax_txt = fig.add_subplot(gs[1])    # Bottom: text
ax_txt.axis('off')

# random portfolios + efficient frontier you computed earlier (tvols/trets)
sc = ax.scatter(pvols, prets, c=(prets - risk_free_rate) / pvols,
                marker='.', cmap='coolwarm', alpha=0.85, label='Random Portfolios')
ax.plot(tvols, trets, 'b', lw=4.0, label='Efficient Frontier')

# CML (guaranteed to pass through ORP)
ax.plot(cml_x, cml_y, 'r', lw=1.8, label='Capital Market Line')

# ORP (Yellow Star with Black Outline)
ax.scatter(orp_vol, orp_ret, s=250, marker='*',
           facecolors='yellow', edgecolors='black',
           linewidths=1.2, zorder=5, label='ORP (Max Sharpe)')

# Individual assets (yellow dots + labels)
ax.scatter(annual_std, annual_ret, s=70, facecolors='yellow', edgecolors='black', zorder=4, label='Assets')
for x, y, name in zip(annual_std.values, annual_ret.values, annual_ret.index):
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6),
                ha='left', fontsize=9, zorder=5)

ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', ls='--', lw=1.2)
ax.axvline(0, color='k', ls='--', lw=1.2)
ax.set_xlabel('expected volatility')
ax.set_ylabel('expected return')
ax.set_title("Optimal Risky Portfolio & CML")
ax.legend(loc='best')

# Text right below the chart
result_text = (
    "================ Optimal Risk Portfolio (Max Sharpe, Long-only) ================\n\n"
    "Assets & Weights (%):\n"
    f"{orp_weights_df.to_string()}\n\n"
    "Summary Stats:\n"
    f"- Volatility (σ):  {orp_vol:.3f}\n"
    f"- Return (μ):      {orp_ret:.3f}\n"
    f"- Sharpe Ratio:    {orp_sr:.3f}   (rf = {risk_free_rate:.2%})\n"
    "===============================================================================\n"
)
ax_txt.text(0.5, 1.0, result_text, ha='center', va='top',
            fontsize=9, family='monospace', transform=ax_txt.transAxes)

plt.show()



##########################################################
### Step 3-A Performance Evaluation of ORP from Step 2 ###
### In-Sample Optimized ORP (Daily Rebalance)          ###
##########################################################

# --- 0) Safety checks / inputs taken from previous steps ---
# Requires: data (Adj Close), risk_free_rate, orp_w, annual_ret
assert 'data' in globals(), "Run Step 0 first (to create 'data')."
assert 'orp_w' in globals() and orp_w is not None, "Run Step 2 first (to compute 'orp_w')."
assert 'annual_ret' in globals(), "Run Steps 0–1 (to compute 'annual_ret')."

# --- 1) Align ORP weights to your price columns ---
w_series = pd.Series(orp_w, index=annual_ret.index)           # ORP weights per ticker from Step 2
w_series = w_series.reindex(data.columns).fillna(0.0)         # match price columns
# Clean tiny numerical noise & renormalize just in case
w_series[w_series.abs() < 1e-12] = 0.0
if not np.isclose(w_series.sum(), 1.0):
    w_series = w_series / w_series.sum()
w_vec = w_series.values

print("\n=== ORP Weights Used for Backtest (% of portfolio) ===")
print((w_series * 100).round(2))

# --- 2) Daily portfolio returns (simple returns) ---
simple_rets = data.pct_change().dropna(how='any')
# If any column has all-NaN after pct_change (rare), drop it & renormalize weights
valid_cols = simple_rets.columns[~simple_rets.isna().all()]
if len(valid_cols) < len(simple_rets.columns):
    dropped = [c for c in simple_rets.columns if c not in valid_cols]
    if dropped:
        print(f"[WARN] Dropping assets with no return data after pct_change: {dropped}")
    simple_rets = simple_rets[valid_cols]
    w_series = w_series.reindex(valid_cols).fillna(0.0)
    if not np.isclose(w_series.sum(), 1.0):
        w_series = w_series / w_series.sum()
    w_vec = w_series.values

port_ret_daily = simple_rets.dot(w_vec)

# --- 3) Build equity curve (index=100 at start) ---
equity = (1.0 + port_ret_daily).cumprod() * 100.0
equity = equity.dropna()

# --- 4) Max Drawdown utility ---
def compute_drawdown(series: pd.Series):
    """
    Returns:
      max_dd: float (e.g., -0.35 for -35%)
      peak_date: Timestamp of the peak before worst drawdown
      trough_date: Timestamp at the drawdown trough
      recovery_date: Timestamp when the series recovers (NaT if never)
      dd_series: full drawdown time series
    """
    running_max = series.cummax()
    dd_series = series / running_max - 1.0

    trough_idx = dd_series.idxmin()
    max_dd = float(dd_series.loc[trough_idx])

    # Peak: last time series was at its running max before trough
    peak_date = series.loc[:trough_idx].idxmax()

    # Recovery: first time after trough that equity exceeds prior peak
    post = series.loc[trough_idx:]
    rec_mask = post.ge(series.loc[peak_date])
    recovery_date = rec_mask.index[rec_mask.argmax()] if rec_mask.any() and post.iloc[rec_mask.argmax()] >= series.loc[peak_date] else pd.NaT

    return max_dd, peak_date, trough_idx, recovery_date, dd_series

max_dd, peak_date, trough_date, recovery_date, dd = compute_drawdown(equity)

print("\n=== Maximum Drawdown (start → end) ===")
print(f"Peak date:     {peak_date.date()}")
print(f"Trough date:   {trough_date.date()}")
print(f"Recovery date: {recovery_date.date() if pd.notna(recovery_date) else 'Not yet recovered'}")
print(f"Max Drawdown:  {max_dd:.2%}")

# --- 5) Performance metrics: CAGR, Annualized Return/Vol, Sharpe ---
trading_days = 252
years = (equity.index[-1] - equity.index[0]).days / 365.25
# Guard for zero/negative years (shouldn't happen if data is valid)
years = max(years, 1e-9)

# CAGR: geometric annual growth
cagr = (equity.iloc[-1] / equity.iloc[0])**(1/years) - 1

# Arithmetic annualized metrics
ann_ret = port_ret_daily.mean() * trading_days
ann_vol = port_ret_daily.std(ddof=0) * math.sqrt(trading_days)  # population stdev for stability
sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan

print("\n=== ORP Portfolio Performance Stats ({} → {}) ===".format(
    equity.index[0].date(), equity.index[-1].date()))
print(f"CAGR:               {cagr:.2%}")
print(f"Annualized Return:  {ann_ret:.2%}")
print(f"Annualized Vol:     {ann_vol:.2%}")
print(f"Sharpe Ratio:       {sharpe:.3f}   (rf = {risk_free_rate:.2%})")

# --- 6) Plot equity curve and drawdown ---
fig = plt.figure(figsize=(9, 6), constrained_layout=True)
gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.05)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# Equity curve
ax1.plot(equity.index, equity.values, lw=1.6)
ax1.set_ylabel('Index Level (start=100)')
ax1.set_title('ORP Portfolio — Cumulative Return')
ax1.grid(True, alpha=0.3)

# Annotate peak & trough
ax1.scatter([peak_date, trough_date],
            [equity.loc[peak_date], equity.loc[trough_date]],
            s=70, edgecolor='black', facecolor='yellow', zorder=5)
ax1.annotate('Peak', xy=(peak_date, equity.loc[peak_date]),
             xytext=(10, 10), textcoords='offset points')
ax1.annotate('Trough', xy=(trough_date, equity.loc[trough_date]),
             xytext=(10, -15), textcoords='offset points')

# Drawdown subplot
ax2.plot(dd.index, dd.values, lw=1.4)
ax2.fill_between(dd.index, dd.values, 0, alpha=0.15)
ax2.set_ylabel('Drawdown')
ax2.set_xlabel('Date')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(min(dd.min() * 1.05, -0.01), 0.02)  # keep 0 near top

plt.show()

# ------------------------------------------------------------------
# (Optional) If you prefer log-return backtest:
# log_rets = np.log(data / data.shift(1)).dropna()
# port_log_daily = log_rets[valid_cols].dot(w_series.values)
# equity_log = np.exp(port_log_daily.cumsum()) * 100.0
# ------------------------------------------------------------------


##########################################################
### Step 3-B Performance Evaluation of ORP from Step 2 ###
### Option 2 ORP (from Step 2) vs Benchmark (BM)       ###
##########################################################

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import yfinance as yf

# ---------- Settings ----------
# Choose one: 'SPY', 'EQUAL_WEIGHT', '60_40'
BM_MODE = 'SPY'     # change here: 'SPY' / 'EQUAL_WEIGHT' / '60_40'

bm_symbols_spy = ['SPY']                    # if BM_MODE == 'SPY'
bm_symbols_6040 = ['SPY', 'TLT']            # if BM_MODE == '60_40'
bm_weights_6040 = np.array([0.60, 0.40])    # 60% SPY / 40% TLT

# ---------- Safety checks ----------
assert 'data' in globals() and isinstance(data, pd.DataFrame), "Run Step 0 first to create 'data' (Adj Close)."
assert 'orp_w' in globals() and orp_w is not None, "Run Step 2 first to compute 'orp_w'."
assert 'risk_free_rate' in globals(), "Define risk_free_rate for Sharpe calculation."

# ---------- Helpers ----------
def compute_drawdown(equity: pd.Series):
    run_max = equity.cummax()
    dd = equity / run_max - 1.0
    return dd

def perf_from_equity(equity: pd.Series, rf: float):
    """Return dict with CAGR, AnnReturn, AnnVol, Sharpe, MaxDD computed from equity curve."""
    equity = equity.dropna()
    trading_days = 252
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    ret_daily = equity.pct_change().dropna()
    ann_ret = ret_daily.mean() * trading_days
    ann_vol = ret_daily.std(ddof=0) * math.sqrt(trading_days)
    sharpe  = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    cagr    = (equity.iloc[-1] / equity.iloc[0])**(1/years) - 1
    dd      = compute_drawdown(equity)
    max_dd  = float(dd.min())
    return {"CAGR": cagr, "AnnRet": ann_ret, "AnnVol": ann_vol, "Sharpe": sharpe, "MaxDD": max_dd}

def pct_fmt(x):
    return f"{x:.2%}" if np.isfinite(x) else "NA"

# ---------- ORP equity (using Step 2 weights over 'data') ----------
rets = data.pct_change().dropna()
w_series = pd.Series(orp_w, index=data.columns).reindex(data.columns).fillna(0.0)
if not np.isclose(w_series.sum(), 1.0):
    w_series = w_series / w_series.sum()

orp_port = rets.dot(w_series.values)
eq_orp = (1.0 + orp_port).cumprod() * 100.0
dd_orp = compute_drawdown(eq_orp)

# ---------- Build Benchmark equity ----------
if BM_MODE == 'SPY':
    if 'SPY' not in data.columns:
        spy_px = yf.download(bm_symbols_spy, start=rets.index[0], end=rets.index[-1], auto_adjust=False)['Adj Close']
        if isinstance(spy_px, pd.Series):
            spy_px = spy_px.to_frame()
        spy_rets = spy_px['SPY'].pct_change().dropna()
    else:
        spy_rets = data['SPY'].pct_change().dropna()
    eq_bm = (1.0 + spy_rets).cumprod() * 100.0
    bm_name = "BM: SPY"

elif BM_MODE == 'EQUAL_WEIGHT':
    n = len(data.columns)
    ew = np.array([1.0/n]*n)
    bm_port = rets.dot(ew)
    eq_bm = (1.0 + bm_port).cumprod() * 100.0
    bm_name = f"BM: Equal-Weight ({n} assets)"

elif BM_MODE == '60_40':
    need = [s for s in bm_symbols_6040 if s not in data.columns]
    if need:
        add_px = yf.download(need, start=rets.index[0], end=rets.index[-1], auto_adjust=False)['Adj Close']
        if isinstance(add_px, pd.Series):
            add_px = add_px.to_frame()
        merged = data.join(add_px, how='outer').sort_index()
    else:
        merged = data.copy()
    merged = merged[bm_symbols_6040].dropna()
    bm_rets = merged.pct_change().dropna()
    w60 = pd.Series(bm_weights_6040, index=bm_symbols_6040).reindex(bm_rets.columns).values
    bm_port = bm_rets.dot(w60)
    eq_bm = (1.0 + bm_port).cumprod() * 100.0
    bm_name = "BM: 60/40 (SPY/TLT)"
else:
    raise ValueError("BM_MODE must be one of: 'SPY', 'EQUAL_WEIGHT', '60_40'.")

# ---------- Align indices & re-index both to 100 at common start ----------
common_idx = eq_orp.index.intersection(eq_bm.index)
eq_orp_c = eq_orp.reindex(common_idx).dropna()
eq_bm_c  = eq_bm.reindex(common_idx).dropna()

# Rebase to 100 at common start
eq_orp_c = eq_orp_c / eq_orp_c.iloc[0] * 100.0
eq_bm_c  = eq_bm_c  / eq_bm_c.iloc[0]  * 100.0

dd_bm = compute_drawdown(eq_bm_c)

# ---------- Print weights + performance summary (ABOVE the charts) ----------
print("\n=== ORP Weights Used for Backtest (% of portfolio) ===")
print((w_series * 100).round(2).rename("Weight %"))

stats_orp = perf_from_equity(eq_orp_c, risk_free_rate)
stats_bm  = perf_from_equity(eq_bm_c,  risk_free_rate)

summary = pd.DataFrame({
    "CAGR":   [stats_orp["CAGR"],  stats_bm["CAGR"]],
    "Ann Ret":[stats_orp["AnnRet"],stats_bm["AnnRet"]],
    "Ann Vol":[stats_orp["AnnVol"],stats_bm["AnnVol"]],
    "Sharpe": [stats_orp["Sharpe"],stats_bm["Sharpe"]],
    "Max DD": [stats_orp["MaxDD"], stats_bm["MaxDD"]],
}, index=["ORP", "BM"])
summary.index.name = "Strategy"

# Format to look like your example (Sharpe also shown as %)
summary_fmt = summary.applymap(pct_fmt)

print(f"\n=== Performance Summary ({common_idx[0].date()} \u2192 {common_idx[-1].date()}) ===")
print(summary_fmt)

# ---------- Plot: Equity (log) + Drawdown ----------
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9, 7),
                               sharex=True, gridspec_kw={'height_ratios':[3,1]})

# Equity (log scale)
ax1.plot(eq_orp_c.index, eq_orp_c.values, lw=1.8, label='ORP (Step 2)')
ax1.plot(eq_bm_c.index,  eq_bm_c.values,  lw=1.8, label=bm_name)
ax1.set_yscale('log')
ax1.set_title("ORP (Step 2) vs Benchmark — Equity (Index=100) [Log Scale]")
ax1.set_ylabel("Index Level (log)")
ax1.grid(True, which='both', alpha=0.3)
ax1.legend(loc='upper left')

# Drawdown (no legend)
ax2.plot(dd_orp.index, dd_orp.values, lw=1.3)       # ORP
ax2.plot(dd_bm.index,  dd_bm.values,  lw=1.3)       # BM
ax2.set_ylabel("Drawdown")
ax2.set_xlabel("Date")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(min(min(dd_orp.min(), dd_bm.min()) * 1.05, -0.01), 0.02)

plt.tight_layout()
plt.show()


##########################################################
### Step 3-C Performance Evaluation of ORP from Step 2 ###
### Buy & Hold vs Daily Rebal vs Monthly Rebal         ###
##########################################################

# --- Safety checks ---
assert 'data' in globals(), "Run Step 0 first (to create 'data')."
assert 'orp_w' in globals() and orp_w is not None, "Run Step 2 first (to compute 'orp_w')."
assert 'annual_ret' in globals(), "Run Steps 0–1 (to compute 'annual_ret')."

# --- Align ORP weights to price columns ---
w_series = pd.Series(orp_w, index=annual_ret.index).reindex(data.columns).fillna(0.0)
w_series[w_series.abs() < 1e-12] = 0.0
if not np.isclose(w_series.sum(), 1.0):
    w_series = w_series / w_series.sum()

print("\n=== ORP Weights Used for Backtest (% of portfolio) ===")
print((w_series * 100).round(2).rename("Weight %"))

# Use fully aligned price panel (drop rows with any missing prices to keep strategies comparable)
prices = data.loc[:, w_series.index].dropna(how='any')
rets   = prices.pct_change()  # simple daily returns (NaN on first row)

# ------------ Utilities ------------
def compute_drawdown(series: pd.Series):
    running_max = series.cummax()
    dd_series = series / running_max - 1.0
    trough_idx = dd_series.idxmin()
    max_dd = float(dd_series.loc[trough_idx])
    peak_date = series.loc[:trough_idx].idxmax()
    post = series.loc[trough_idx:]
    rec_mask = post.ge(series.loc[peak_date])
    recovery_date = (rec_mask.index[rec_mask.argmax()]
                     if rec_mask.any() and post.iloc[rec_mask.argmax()] >= series.loc[peak_date]
                     else pd.NaT)
    return max_dd, peak_date, trough_idx, recovery_date, dd_series

def perf_from_equity(equity: pd.Series, rf: float):
    trading_days = 252
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    ret_daily = equity.pct_change().dropna()
    ann_ret = ret_daily.mean() * trading_days
    ann_vol = ret_daily.std(ddof=0) * math.sqrt(trading_days)
    cagr    = (equity.iloc[-1] / equity.iloc[0])**(1/years) - 1
    sharpe  = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    max_dd, peak, trough, rec, dd = compute_drawdown(equity)
    return {
        "CAGR": cagr, "Ann_Return": ann_ret, "Ann_Vol": ann_vol,
        "Sharpe": sharpe, "MaxDD": max_dd,
        "Peak": peak, "Trough": trough, "Recovery": rec, "DD_Series": dd
    }

# ------------ 1) Buy & Hold (no rebalance) ------------
init_wealth = 100.0
shares_bh = (init_wealth * w_series) / prices.iloc[0]
equity_bh = prices.mul(shares_bh, axis=1).sum(axis=1)
equity_bh = equity_bh / equity_bh.iloc[0] * 100.0  # start at 100

# ------------ 2) Daily Rebalance (constant-mix) ------------
daily_port_r = rets.dot(w_series.values).fillna(0.0)
equity_daily = (1.0 + daily_port_r).cumprod() * 100.0  # start at 100

# ------------ 3) Monthly Rebalance ------------
wealth = 100.0
equity_month = pd.Series(index=prices.index, dtype=float)

# first trading day of each calendar month
rebal_dates = set(prices.groupby(prices.index.to_period('M')).apply(lambda df: df.index[0]).tolist())
shares = None
last_rebal = None

for t in prices.index:
    # rebalance at first trading day of each month (and at the very first date)
    if (last_rebal is None) or (t in rebal_dates and t != last_rebal):
        shares = (wealth * w_series) / prices.loc[t]
        last_rebal = t
    wealth = float((prices.loc[t] * shares).sum())
    equity_month.loc[t] = wealth

equity_month = equity_month / equity_month.iloc[0] * 100.0  # start at 100

# ------------ Performance stats ------------
stats_bh     = perf_from_equity(equity_bh,     risk_free_rate)
stats_daily  = perf_from_equity(equity_daily,  risk_free_rate)
stats_month  = perf_from_equity(equity_month,  risk_free_rate)

# Print summary table
summary = pd.DataFrame({
    "Strategy": ["Buy & Hold", "Daily Rebalance", "Monthly Rebalance"],
    "CAGR":     [stats_bh["CAGR"],    stats_daily["CAGR"],    stats_month["CAGR"]],
    "Ann Ret":  [stats_bh["Ann_Return"], stats_daily["Ann_Return"], stats_month["Ann_Return"]],
    "Ann Vol":  [stats_bh["Ann_Vol"], stats_daily["Ann_Vol"], stats_month["Ann_Vol"]],
    "Sharpe":   [stats_bh["Sharpe"],  stats_daily["Sharpe"],  stats_month["Sharpe"]],
    "Max DD":   [stats_bh["MaxDD"],   stats_daily["MaxDD"],   stats_month["MaxDD"]],
})
print("\n=== Performance Summary ({} → {}) ===".format(prices.index[0].date(), prices.index[-1].date()))
print(summary.set_index("Strategy").applymap(lambda x: f"{x:.2%}" if isinstance(x, (float, np.floating)) else x))

# ------------ Figure 1: Equity curves (three lines) ------------
plt.figure(figsize=(10,6))
plt.plot(equity_bh.index,    equity_bh.values,    label="Buy & Hold (No Rebalance)", lw=1.8)
plt.plot(equity_daily.index, equity_daily.values, label="Daily Rebalance", lw=1.8)
plt.plot(equity_month.index, equity_month.values, label="Monthly Rebalance", lw=1.8)
plt.title("ORP Portfolio — Equity Curves (All Start at 100)")
plt.ylabel("Index Level (start=100)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ------------ Figure 2: Drawdowns (one chart, 3 lines) ------------
dd_bh    = stats_bh["DD_Series"]
dd_daily = stats_daily["DD_Series"]
dd_month = stats_month["DD_Series"]

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(dd_bh.index,    dd_bh.values,    lw=1.6, label=f"Buy & Hold (Max DD {stats_bh['MaxDD']:.1%})")
ax.plot(dd_daily.index, dd_daily.values, lw=1.6, label=f"Daily Rebalance (Max DD {stats_daily['MaxDD']:.1%})")
ax.plot(dd_month.index, dd_month.values, lw=1.6, label=f"Monthly Rebalance (Max DD {stats_month['MaxDD']:.1%})")

# mark trough points for each strategy
for dd_series, stats in [(dd_bh, stats_bh), (dd_daily, stats_daily), (dd_month, stats_month)]:
    t = stats["Trough"]
    ax.scatter(t, dd_series.loc[t], s=70, edgecolor="black", facecolor="yellow", zorder=5)

ax.axhline(0, color='k', lw=1.0, ls='--')
ax.set_title("Drawdowns — All Strategies")
ax.set_ylabel("Drawdown")
ax.set_xlabel("Date")
ax.grid(True, alpha=0.3)

# unified y-limits
ymin = min(dd_bh.min(), dd_daily.min(), dd_month.min()) * 1.05
ax.set_ylim(min(ymin, -0.01), 0.02)

ax.legend()
plt.tight_layout()
plt.show()


#########################################################
### Step 4  User-Defined Portfolios (Fixed Weights)   ###
###         ORP and 36M Rolling ORP (Monthly Rebal    ###
#########################################################

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.optimize as sco

# ===== Preconditions (from Step 0/2) =====
assert 'start_date'     in globals(), "Define start_date (e.g., '2005-01-01')"
assert 'end_date'       in globals(), "Define end_date   (e.g., '2025-08-31')"
assert 'risk_free_rate' in globals(), "Define risk_free_rate (e.g., 0.02)"
# If Step 0/2 already created 'data' or 'orp_w', they are not required here. This file is self-contained.

# ===== Minimal helpers (self-contained) =====
def load_prices(symbols, start_date, end_date):
    px = yf.download(symbols, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
    if isinstance(px, pd.Series): px = px.to_frame()
    cols_in = [s for s in symbols if s in px.columns]
    px = px.loc[:, cols_in].dropna(how='all').dropna()
    return px

def annualize_from_simple(rets, ddof_std=0):
    ann_ret  = rets.mean() * 252
    ann_std  = rets.std(ddof=ddof_std) * math.sqrt(252)
    ann_cov  = rets.cov(ddof=ddof_std) * 252
    ann_corr = rets.corr()
    return ann_ret, ann_std, ann_cov, ann_corr

def orp_weights_from_annual(ann_ret, ann_cov, rf):
    def port_ret(w): return float(w @ ann_ret.values)
    def port_vol(w): return float((w @ ann_cov.values @ w) ** 0.5)
    def neg_sharpe(w):
        v = port_vol(w)
        return -((port_ret(w) - rf) / v) if v > 0 else 1e9
    n = len(ann_ret)
    bnds = tuple((0,1) for _ in range(n))
    cons = ({'type':'eq','fun': lambda x: np.sum(x) - 1},)
    w0   = np.array([1.0/n]*n)
    res  = sco.minimize(neg_sharpe, w0, method='SLSQP', bounds=bnds, constraints=cons)
    w    = res.x
    w[np.abs(w) < 1e-12] = 0.0
    if not np.isclose(w.sum(), 1.0):
        w = w / w.sum()
    return pd.Series(w, index=ann_ret.index)

# ===== Backtest utilities =====
def backtest_from_weights_monthly(prices, weights_s, rf):
    """
    Fixed weights re-applied at the start of each calendar month.
    Returns: equity (index=100), drawdown series, and a stats dict.
    """
    weights_s = weights_s.reindex(prices.columns).fillna(0.0)
    if not np.isclose(weights_s.sum(), 1.0):
        weights_s = weights_s / weights_s.sum()
    w    = weights_s.values
    rets = prices.pct_change().dropna()

    # Keep the same weights within each calendar month
    parts = []
    for _, g in rets.groupby(rets.index.to_period('M')):
        parts.append(g.dot(w))
    port = pd.concat(parts).sort_index()

    # Equity and drawdown
    eq = (1.0 + port).cumprod() * 100.0
    run_max = eq.cummax()
    dd = eq / run_max - 1.0
    trough = dd.idxmin()
    max_dd = float(dd.loc[trough])
    peak   = eq.loc[:trough].idxmax()
    post   = eq.loc[trough:]
    rec_m  = post.ge(eq.loc[peak])
    recovery = (rec_m.index[rec_m.argmax()]
                if rec_m.any() and post.iloc[rec_m.argmax()] >= eq.loc[peak] else pd.NaT)

    # Daily-based annualization (arithmetic)
    td   = 252
    annR = port.mean() * td
    annV = port.std(ddof=0) * math.sqrt(td)
    sharpe = (annR - rf) / annV if annV > 0 else np.nan
    years  = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr   = (eq.iloc[-1] / eq.iloc[0])**(1/years) - 1

    stats = {"CAGR": cagr, "AnnReturn": annR, "AnnVol": annV, "Sharpe": sharpe,
             "MaxDD": max_dd, "PeakDate": peak, "TroughDate": trough, "RecoveryDate": recovery}
    return eq, dd, stats

def rolling_orp_backtest_monthly(prices, rf, window_months=36):
    """
    Out-of-sample rolling ORP:
      - For each month t, optimize ORP using prior `window_months` of DAILY returns.
      - Hold those weights within month t (monthly rebalancing).
    """
    rets = prices.pct_change().dropna()
    mlab = rets.index.to_period('M')
    um   = mlab.unique().sort_values()

    out_parts = []
    for i in range(window_months, len(um)):
        train_m = um[i-window_months:i]
        this_m  = um[i]
        train   = rets[mlab.isin(train_m)]
        if train.shape[0] < 60:  # minimal guard for too-short sample
            continue
        annR, _, annC, _ = annualize_from_simple(train, ddof_std=0)
        w = orp_weights_from_annual(annR, annC, rf).reindex(prices.columns).fillna(0.0)
        if not np.isclose(w.sum(), 1.0): w = w / max(w.sum(), 1e-12)
        this = rets[mlab == this_m]
        if this.empty: continue
        out_parts.append(this.dot(w.values))

    if not out_parts:
        raise ValueError("Not enough data for rolling ORP. Extend the date range or reduce window.")

    port = pd.concat(out_parts).sort_index()
    eq   = (1.0 + port).cumprod() * 100.0
    run_max = eq.cummax()
    dd   = eq / run_max - 1.0
    trough = dd.idxmin()
    max_dd = float(dd.loc[trough])
    peak   = eq.loc[:trough].idxmax()
    post   = eq.loc[trough:]
    rec_m  = post.ge(eq.loc[peak])
    recovery = (rec_m.index[rec_m.argmax()]
                if rec_m.any() and post.iloc[rec_m.argmax()] >= eq.loc[peak] else pd.NaT)

    td   = 252
    annRet = port.mean() * td
    annVol = port.std(ddof=0) * math.sqrt(td)
    sharpe = (annRet - rf) / annVol if annVol > 0 else np.nan
    years  = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr   = (eq.iloc[-1] / eq.iloc[0])**(1/years) - 1

    stats = {"CAGR": cagr, "AnnReturn": annRet, "AnnVol": annVol, "Sharpe": sharpe,
             "MaxDD": max_dd, "PeakDate": peak, "TroughDate": trough, "RecoveryDate": recovery}
    return eq, dd, stats

def normalize_and_filter_weights(weights_dict, available_cols):
    """
    Keep only tickers present in `available_cols`; remove NaNs/zeros; normalize to 1.
    Returns a pd.Series aligned to `available_cols` (missing tickers become 0).
    """
    w = pd.Series(weights_dict, dtype=float)
    w = w[w.index.isin(available_cols)].replace([np.inf, -np.inf], np.nan).dropna()
    w = w[w != 0]
    s = pd.Series(0.0, index=available_cols, dtype=float)
    if len(w) == 0: return s
    w = w / w.sum()
    s.loc[w.index] = w.values
    return s

# ===== (1) User input: strategy name + weights =====
# Add as many user strategies as you want.
user_weight_sets = [
    ("All Weather (User Weights)", {
        "SPY": 0.30, "TLT": 0.40, "IEF": 0.15, "GLD": 0.15
    }),
    # ("My Custom Mix", {"SPY":0.5, "GLD":0.2, "TLT":0.3})
]

include_orp_benchmarks = True  # Also include ORP (in-sample) on the same universe (monthly rebal)

# ===== (2) Build: fixed weights (monthly) + optional ORP (monthly) =====
def build_fixed_and_optional_orp_monthly(name, weights_dict, start_date, end_date, rf):
    tickers = sorted(list(weights_dict.keys()))
    px = load_prices(tickers, start_date, end_date)
    if px.shape[1] == 0:
        raise ValueError("No valid price columns found for provided weights.")

    # Fixed (monthly)
    w_fixed = normalize_and_filter_weights(weights_dict, px.columns)
    if np.isclose(w_fixed.sum(), 0.0):
        raise ValueError("After filtering, provided weights are all zero or invalid.")
    eq_fixed, dd_fixed, stats_fixed = backtest_from_weights_monthly(px, w_fixed, rf)

    out = [{
        "name": name,
        "type": "Fixed (Monthly Rebal)",
        "prices": px, "weights": w_fixed,
        "equity": eq_fixed, "drawdown": dd_fixed, "stats": stats_fixed
    }]

    # ORP (in-sample, monthly)
    if include_orp_benchmarks:
        rets = px.pct_change().dropna()
        annR, _, annC, _ = annualize_from_simple(rets, ddof_std=0)
        w_orp = orp_weights_from_annual(annR, annC, rf)
        eq_orp, dd_orp, stats_orp = backtest_from_weights_monthly(px, w_orp, rf)
        orp_name = name.replace("(User Weights)", "(ORP Benchmark)")

        out.append({
            "name": orp_name,
            "type": "ORP (Monthly Rebal)",
            "prices": px, "weights": w_orp,
            "equity": eq_orp, "drawdown": dd_orp, "stats": stats_orp
        })
    return out

# ===== (3) Build all user strategies =====
results_user = []
for name, wdict in user_weight_sets:
    try:
        bundles = build_fixed_and_optional_orp_monthly(name, wdict, start_date, end_date, risk_free_rate)
        results_user.extend(bundles)
    except Exception as e:
        print(f"[WARN] Skipped '{name}' due to: {e}")

assert len(results_user) >= 1, "No user-defined portfolios were built. Check tickers/weights/date range."

# ===== (4) Add ORP Rolling 36m (OOS) on the same universe as the first user strategy =====
base_px = results_user[0]["prices"]
eq_roll, dd_roll, stats_roll = rolling_orp_backtest_monthly(base_px, risk_free_rate, window_months=36)
results_user.append({
    "name": results_user[0]["name"].replace("(User Weights)", "(ORP Rolling 36m)"),
    "type": "ORP (Rolling 36m, Monthly Rebal)",
    "prices": base_px, "weights": None,
    "equity": eq_roll, "drawdown": dd_roll, "stats": stats_roll
})

# ===== (5) Charts: log equity + drawdown =====
# Rebase all series to 100 at the common start date (intersection of all series)
common_idx = results_user[0]["equity"].index
for r in results_user[1:]:
    common_idx = common_idx.intersection(r["equity"].index)

equity_df = pd.DataFrame({r["name"]: r["equity"].reindex(common_idx) for r in results_user}).dropna()
dd_df     = pd.DataFrame({r["name"]: r["drawdown"].reindex(common_idx) for r in results_user}).dropna()
equity_df = equity_df / equity_df.iloc[0] * 100.0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={'height_ratios':[3,1]})
equity_df.plot(ax=ax1, lw=1.8)
ax1.set_yscale('log')
ax1.set_title("User Weights vs ORP (In-sample) vs ORP (Rolling 36m) — Monthly Rebal")
ax1.set_ylabel("Index Level (log, start=100)")
ax1.grid(True, which='both', alpha=0.3)
ax1.legend(loc='upper left')

dd_df.plot(ax=ax2, lw=1.2, legend=False)
ax2.set_ylabel("Drawdown")
ax2.set_xlabel("Date")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(min(dd_df.min().min()*1.05, -0.01), 0.02)
plt.tight_layout()
plt.show()

# ===== (6) Performance matrix (AnnReturn, AnnVol, Sharpe, MaxDD, CAGR) =====
def pct(x):  return f"{x:.2%}" if np.isfinite(x) else "NA"
def num(x):  return float(x) if np.isfinite(x) else np.nan

rows_num = []
rows_fmt = []
for r in results_user:
    s = r["stats"]
    rows_num.append({
        "Portfolio": r["name"],
        "AnnReturn": num(s["AnnReturn"]),
        "AnnVol":    num(s["AnnVol"]),
        "Sharpe":    num(s["Sharpe"]),
        "MaxDD":     num(s["MaxDD"]),
        "CAGR":      num(s["CAGR"])
    })
    rows_fmt.append({
        "Portfolio": r["name"],
        "Ann. Return": pct(s["AnnReturn"]),
        "Ann. Vol":    pct(s["AnnVol"]),
        "Sharpe":      f"{s['Sharpe']:.3f}" if np.isfinite(s["Sharpe"]) else "NA",
        "MaxDD":       pct(s["MaxDD"]),
        "CAGR":        pct(s["CAGR"])
    })

perf_matrix_numeric = pd.DataFrame(rows_num).set_index("Portfolio")
perf_matrix_display = pd.DataFrame(rows_fmt).set_index("Portfolio")

print("\n==================== Performance Matrix (numeric) ====================")
print(perf_matrix_numeric.round(4))
print("======================================================================\n")

print("==================== Performance Matrix (formatted) ===================")
print(perf_matrix_display)
print("=======================================================================\n")

# ===== (7) Notes =====
# - Add more user strategies in `user_weight_sets`.
# - Labels automatically pair as: "(User Weights)" → "(ORP Benchmark)" → "(ORP Rolling 36m)".
# - All curves are rebased to 100 at the common start; top: log-scale equity; bottom: drawdowns.


