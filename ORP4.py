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

#########################################################
### Step 3  Performance Evaluation                    ###
###        (equity curve, drawdown, CAGR, Sharpe)     ###
#########################################################

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
simple_rets = data.pct_change().dropna(how='all')
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


#########################################################
### Step 4  Compare Performance Across ORPs           ###
###        (Composed of different assets)             ###
###        (Equity Curves + Drawdowns + Stats)        ###
#########################################################

# Assumes you already defined: risk_free_rate, start_date, end_date
# and imported: pandas as pd, numpy as np, math, yfinance as yf, matplotlib.pyplot as plt, scipy.optimize as sco

# ---------- Helpers ----------
def load_prices(symbols, start_date, end_date):
    """Download Adj Close prices and return a clean DataFrame."""
    px = yf.download(symbols, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
    if isinstance(px, pd.Series):
        px = px.to_frame()
    cols_in = [s for s in symbols if s in px.columns]
    px = px.loc[:, cols_in].dropna(how='all').dropna()
    return px

def annualize_from_simple(rets, ddof_std=0):
    """Annualized mean/vol/cov/corr from daily simple returns."""
    ann_ret  = rets.mean() * 252
    ann_std  = rets.std(ddof=ddof_std) * math.sqrt(252)
    ann_cov  = rets.cov(ddof=ddof_std) * 252
    ann_corr = rets.corr()
    return ann_ret, ann_std, ann_cov, ann_corr

def orp_weights_from_annual(ann_ret, ann_cov, rf):
    """Compute long-only ORP (Max Sharpe) weights using SLSQP."""
    def port_ret(w): return float(w @ ann_ret.values)
    def port_vol(w): return float((w @ ann_cov.values @ w) ** 0.5)
    def neg_sharpe(w):
        vol = port_vol(w)
        return -((port_ret(w) - rf) / vol) if vol > 0 else 1e9

    n = len(ann_ret)
    bnds = tuple((0, 1) for _ in range(n))
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    w0 = np.array([1.0/n]*n)
    res = sco.minimize(neg_sharpe, w0, method='SLSQP', bounds=bnds, constraints=cons)
    w = res.x
    w[np.abs(w) < 1e-12] = 0.0
    if not np.isclose(w.sum(), 1.0):
        w = w / w.sum()
    return pd.Series(w, index=ann_ret.index)

def backtest_from_weights(prices, weights_s, rf):
    """Backtest daily simple returns → equity, drawdown, and stats dict."""
    weights_s = weights_s.reindex(prices.columns).fillna(0.0)
    if not np.isclose(weights_s.sum(), 1.0):
        weights_s = weights_s / weights_s.sum()
    w = weights_s.values

    # Daily portfolio simple returns
    rets = prices.pct_change().dropna()
    port = rets.dot(w)

    # Equity curve (index=100)
    eq = (1.0 + port).cumprod() * 100.0

    # Drawdown series
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    trough_date = dd.idxmin()
    max_dd = float(dd.loc[trough_date])
    peak_date = eq.loc[:trough_date].idxmax()
    post = eq.loc[trough_date:]
    rec_mask = post.ge(eq.loc[peak_date])
    recovery_date = rec_mask.index[rec_mask.argmax()] if rec_mask.any() and post.iloc[rec_mask.argmax()] >= eq.loc[peak_date] else pd.NaT

    # Annualized metrics from daily arithmetic
    trading_days = 252
    ann_ret = port.mean() * trading_days
    ann_vol = port.std(ddof=0) * math.sqrt(trading_days)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan

    # CAGR (geometric)
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0])**(1/years) - 1

    stats = {
        "CAGR": cagr,
        "AnnReturn": ann_ret,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "PeakDate": peak_date,
        "TroughDate": trough_date,
        "RecoveryDate": recovery_date
    }
    return eq, dd, stats

def build_orp_and_backtest(name, symbols, start_date, end_date, rf):
    """Full pipeline for one portfolio set."""
    px   = load_prices(symbols, start_date, end_date)
    rets = px.pct_change().dropna()
    ann_ret, ann_std, ann_cov, _ = annualize_from_simple(rets, ddof_std=0)
    w_orp = orp_weights_from_annual(ann_ret, ann_cov, rf)
    eq, dd, stats = backtest_from_weights(px, w_orp, rf)
    return {
        "name": name,
        "symbols": symbols,
        "prices": px,
        "weights": w_orp,
        "equity": eq,
        "drawdown": dd,
        "stats": stats
    }

# ---------- Define the multiple portfolios ----------
sets = [
    ("Equity Index (SPY/FXI/EWQ/EWG)", ['SPY', 'FXI', 'EWQ', 'EWG']),
    ("All Weather (SPY/TLT/IEF/GLD)", ['SPY','TLT','IEF','GLD'])
]
# ("Stock Pick Genius (GLD/SPY/MSFT/AAPL)", ['GLD','SPY','MSFT','AAPL']),

# ---------- Build each ORP & backtest ----------
results = []
for name, syms in sets:
    try:
        res = build_orp_and_backtest(name, syms, start_date, end_date, risk_free_rate)
        results.append(res)
    except Exception as e:
        print(f"[WARN] Skipped {name} due to error: {e}")

assert len(results) >= 1, "No portfolios were successfully built. Check tickers/date range/network."

# ---------- Align equity & drawdown on a common date range ----------
common_index = results[0]["equity"].index
for r in results[1:]:
    common_index = common_index.intersection(r["equity"].index)

equity_df = pd.DataFrame({ r["name"]: r["equity"].reindex(common_index) for r in results }).dropna()
dd_df     = pd.DataFrame({ r["name"]: r["drawdown"].reindex(common_index) for r in results }).dropna()

# ---------- Plot combined equity curves (LOG) + Drawdowns ----------
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9, 7),
                               sharex=True, gridspec_kw={'height_ratios':[3,1]})

# Equity: log scale with legend
equity_df.plot(ax=ax1, lw=1.8)
ax1.set_yscale('log')
ax1.set_title("ORP Equity Curves (Index = 100 at Common Start) — Log Scale")
ax1.set_ylabel("Index Level (log scale)")
ax1.grid(True, which='both', axis='both', alpha=0.3)
ax1.legend(loc='upper left')

# Drawdowns: one line per portfolio (no labels/legend)
dd_df.plot(ax=ax2, lw=1.2, legend=False)
ax2.set_ylabel("Drawdown")
ax2.set_xlabel("Date")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(min(dd_df.min().min() * 1.05, -0.01), 0.02)

plt.tight_layout()
plt.show()

# ---------- Comparison table (CAGR, AnnReturn, AnnVol, Sharpe, MaxDD) ----------
def pct(x): 
    return f"{x:.2%}" if np.isfinite(x) else "NA"

rows = []
for r in results:
    s = r["stats"]
    rows.append({
        "Portfolio": r["name"],
        "CAGR": pct(s["CAGR"]),
        "Ann. Return": pct(s["AnnReturn"]),
        "Ann. Vol": pct(s["AnnVol"]),
        "Sharpe".format(risk_free_rate): f"{s['Sharpe']:.3f}" if np.isfinite(s["Sharpe"]) else "NA",
        "Max Drawdown": pct(s["MaxDD"])
    })
comparison = pd.DataFrame(rows).set_index("Portfolio")
print("\n==================== ORP Performance Comparison ====================")
print(comparison)
print("====================================================================\n")

# ---------- Show ORP weights for each portfolio ----------
for r in results:
    w = (r["weights"] * 100).round(2).sort_values(ascending=False)
    print(f"--- ORP Weights (%): {r['name']} ---")
    print(w.to_string())
    print()


#########################################################
### Step 5  User-Defined Portfolios (Fixed Weights)   ###
###         Monthly Rebalancing + Comparison          ###
#########################################################

# -- Monthly-rebalanced backtest for FIXED weights --
def backtest_from_weights_monthly(prices, weights_s, rf):
    """
    Fixed weights re-applied at the start of each calendar month.
    Returns daily equity & drawdown series and stats.
    """
    weights_s = weights_s.reindex(prices.columns).fillna(0.0)
    if not np.isclose(weights_s.sum(), 1.0):
        weights_s = weights_s / weights_s.sum()
    w = weights_s.values

    rets = prices.pct_change().dropna()

    # Same weights within each month
    port_daily_parts = []
    for _, g in rets.groupby(rets.index.to_period('M')):
        port_daily_parts.append(g.dot(w))
    port = pd.concat(port_daily_parts).sort_index()

    # Equity curve (index=100)
    eq = (1.0 + port).cumprod() * 100.0

    # Drawdowns
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    trough_date = dd.idxmin()
    max_dd = float(dd.loc[trough_date])
    peak_date = eq.loc[:trough_date].idxmax()
    post = eq.loc[trough_date:]
    rec_mask = post.ge(eq.loc[peak_date])
    recovery_date = rec_mask.index[rec_mask.argmax()] if rec_mask.any() and post.iloc[rec_mask.argmax()] >= eq.loc[peak_date] else pd.NaT

    # Daily-based annualization
    trading_days = 252
    ann_ret = port.mean() * trading_days
    ann_vol = port.std(ddof=0) * math.sqrt(trading_days)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan

    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0])**(1/years) - 1

    stats = {
        "CAGR": cagr,
        "AnnReturn": ann_ret,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "PeakDate": peak_date,
        "TroughDate": trough_date,
        "RecoveryDate": recovery_date
    }
    return eq, dd, stats


# -- 36-month rolling ORP, out-of-sample (monthly rebal) --
def rolling_orp_backtest_monthly(prices, rf, window_months=36):
    """
    Out-of-sample ORP:
      - For each month t, optimize ORP using prior `window_months` of DAILY returns.
      - Hold those weights within month t (monthly rebalancing).
      - Returns daily equity/drawdown and stats.
    """
    rets = prices.pct_change().dropna()
    month_labels = rets.index.to_period('M')
    unique_months = month_labels.unique().sort_values()

    port_series_list = []
    for i in range(window_months, len(unique_months)):
        train_months = unique_months[i - window_months : i]
        this_month   = unique_months[i]

        train_rets = rets[month_labels.isin(train_months)]
        if train_rets.shape[0] < 60:
            continue

        ann_ret, ann_std, ann_cov, _ = annualize_from_simple(train_rets, ddof_std=0)
        w_orp = orp_weights_from_annual(ann_ret, ann_cov, rf).reindex(prices.columns).fillna(0.0)
        if not np.isclose(w_orp.sum(), 1.0):
            w_orp = w_orp / max(w_orp.sum(), 1e-12)

        month_rets = rets[month_labels == this_month]
        if month_rets.empty:
            continue
        port_series_list.append(month_rets.dot(w_orp.values))

    if not port_series_list:
        raise ValueError("Not enough data for rolling ORP. Extend date range or reduce window.")

    port = pd.concat(port_series_list).sort_index()

    # Equity / Drawdown / Stats
    eq = (1.0 + port).cumprod() * 100.0
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    trough_date = dd.idxmin()
    max_dd = float(dd.loc[trough_date])
    peak_date = eq.loc[:trough_date].idxmax()
    post = eq.loc[trough_date:]
    rec_mask = post.ge(eq.loc[peak_date])
    recovery_date = rec_mask.index[rec_mask.argmax()] if rec_mask.any() and post.iloc[rec_mask.argmax()] >= eq.loc[peak_date] else pd.NaT

    trading_days = 252
    ann_return = port.mean() * trading_days
    ann_vol    = port.std(ddof=0) * math.sqrt(trading_days)
    sharpe     = (ann_return - rf) / ann_vol if ann_vol > 0 else np.nan

    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr  = (eq.iloc[-1] / eq.iloc[0])**(1/years) - 1

    stats = {
        "CAGR": cagr,
        "AnnReturn": ann_return,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "PeakDate": peak_date,
        "TroughDate": trough_date,
        "RecoveryDate": recovery_date
    }
    return eq, dd, stats


# -- Define your user portfolios & weights (edit here) --
user_weight_sets = [
    ("All Weather (User Weights)", {
        "SPY": 0.30, "TLT": 0.40, "IEF": 0.15, "GLD": 0.15
    }),
    # ("My Custom Mix", {"SPY":0.5, "GLD":0.2, "TLT":0.3})
]

# Also include an ORP benchmark on the same universe, monthly rebalanced
include_orp_benchmarks = True   # set False if you only want your fixed mixes


# -- Helpers specific to Step 5 --
def normalize_and_filter_weights(weights_dict, available_cols):
    """
    Keep only tickers present in available_cols; drop zeros/NaNs; normalize to 1.
    Returns a pd.Series aligned to available_cols (missing tickers = 0).
    """
    w = pd.Series(weights_dict, dtype=float)
    w = w[w.index.isin(available_cols)].replace([np.inf, -np.inf], np.nan).dropna()
    w = w[w != 0]
    s = pd.Series(0.0, index=available_cols, dtype=float)
    if len(w) == 0:
        return s
    w = w / w.sum()
    s.loc[w.index] = w.values
    return s


def build_fixed_and_optional_orp_monthly(name, weights_dict, start_date, end_date, rf):
    """
    Downloads universe, backtests FIXED weights with MONTHLY rebalancing.
    Optionally also builds the MONTHLY-rebalanced ORP benchmark on same universe.
    """
    tickers = sorted(list(weights_dict.keys()))
    px = load_prices(tickers, start_date, end_date)
    if px.shape[1] == 0:
        raise ValueError("No valid price columns found for provided weights.")

    # Fixed weights aligned to px
    w_fixed = normalize_and_filter_weights(weights_dict, px.columns)
    if np.isclose(w_fixed.sum(), 0.0):
        raise ValueError("After filtering, provided weights are all zero or invalid.")

    # Backtest FIXED (monthly rebal)
    eq_fixed, dd_fixed, stats_fixed = backtest_from_weights_monthly(px, w_fixed, rf)

    results_list = [{
        "name": name,
        "type": "Fixed (Monthly Rebal)",
        "prices": px,
        "weights": w_fixed,
        "equity": eq_fixed,
        "drawdown": dd_fixed,
        "stats": stats_fixed
    }]

    if include_orp_benchmarks:
        # In-sample ORP on same universe
        rets = px.pct_change().dropna()
        ann_ret, ann_std, ann_cov, _ = annualize_from_simple(rets, ddof_std=0)
        w_orp = orp_weights_from_annual(ann_ret, ann_cov, rf)
        eq_orp, dd_orp, stats_orp = backtest_from_weights_monthly(px, w_orp, rf)

        # Label fix: "All Weather (ORP Benchmark)"
        orp_name = name.replace("(User Weights)", "(ORP Benchmark)")

        results_list.append({
            "name": orp_name,
            "type": "ORP (Monthly Rebal)",
            "prices": px,
            "weights": w_orp,
            "equity": eq_orp,
            "drawdown": dd_orp,
            "stats": stats_orp
        })
    return results_list


# -- Build all requested user portfolios (monthly) --
results_user = []
for name, wdict in user_weight_sets:
    try:
        bundles = build_fixed_and_optional_orp_monthly(name, wdict, start_date, end_date, risk_free_rate)
        results_user.extend(bundles)
    except Exception as e:
        print(f"[WARN] Skipped '{name}' due to: {e}")

assert len(results_user) >= 1, "No user-defined portfolios were built. Check tickers/weights/date range."

# --- Add 36m rolling ORP for the same All Weather universe ---
aw_result = results_user[0]  # "All Weather (User Weights)"
aw_px = aw_result["prices"]
eq_roll, dd_roll, stats_roll = rolling_orp_backtest_monthly(aw_px, risk_free_rate, window_months=36)
results_user.append({
    "name": "All Weather (ORP Rolling 36m)",
    "type": "ORP (Rolling 36m, Monthly Rebal)",
    "prices": aw_px,
    "weights": None,   # time-varying; omitted
    "equity": eq_roll,
    "drawdown": dd_roll,
    "stats": stats_roll
})


# -- Align curves & plot (log equity + drawdown) --
# Intersection ensures all series share the SAME start (so rebasing aligns at 100)
common_index_u = results_user[0]["equity"].index
for r in results_user[1:]:
    common_index_u = common_index_u.intersection(r["equity"].index)

equity_user_df = pd.DataFrame({ r["name"]: r["equity"].reindex(common_index_u) for r in results_user }).dropna()
dd_user_df     = pd.DataFrame({ r["name"]: r["drawdown"].reindex(common_index_u) for r in results_user }).dropna()

# Rebase every series to 100 at the common start
equity_user_df = equity_user_df / equity_user_df.iloc[0] * 100.0

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9, 7),
                               sharex=True, gridspec_kw={'height_ratios':[3,1]})

# Equity curves (log scale)
equity_user_df.plot(ax=ax1, lw=1.8)
ax1.set_yscale('log')
ax1.set_title("User-Defined Portfolio(s) — Equity Curves (Index=100 @ Common Start) — Monthly Rebal")
ax1.set_ylabel("Index Level (log)")
ax1.grid(True, which='both', axis='both', alpha=0.3)
ax1.legend(loc='upper left')

# Drawdowns (no legend)
dd_user_df.plot(ax=ax2, lw=1.2, legend=False)
ax2.set_ylabel("Drawdown")
ax2.set_xlabel("Date")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(min(dd_user_df.min().min() * 1.05, -0.01), 0.02)

plt.tight_layout()
plt.show()


# -- Comparison table (compact, fixed-width; uses actual names) --
def pct(x): 
    return f"{x:.2%}" if np.isfinite(x) else "NA"

# Optional helper to avoid name overflow (ellipsis)
def fit_cell(text, width):
    s = str(text)
    return s if len(s) <= width else (s[:max(0, width-1)] + "…")

rows = []
for r in results_user:
    s = r["stats"]
    rows.append([
        r["name"],  # actual portfolio name
        pct(s["CAGR"]),
        pct(s["AnnReturn"]),
        pct(s["AnnVol"]),
        f"{s['Sharpe']:.3f}" if np.isfinite(s["Sharpe"]) else "NA",
        pct(s["MaxDD"]),
        s["RecoveryDate"].date().strftime("%Y-%m-%d") if pd.notna(s["RecoveryDate"]) else "Not yet"
    ])

headers    = ["Portfolio", "CAGR", "Ann. Return", "Ann. Vol", "Sharpe", "MaxDD", "Recovery"]
col_widths = [32,          8,       12,            10,        8,       10,       12]

header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
sep_line    = "-" * sum(col_widths)

print("\n==================== User vs ORP (In-sample & Rolling 36m) — Monthly Rebal ====================")
print(header_line)
print(sep_line)
for row in rows:
    cells = [fit_cell(val, w).ljust(w) for val, w in zip(row, col_widths)]
    print("".join(cells))
print("================================================================================================\n")

# -- Show actual weights used (rolling ORP has time-varying weights, so skip) --
for r in results_user:
    if r["weights"] is None:
        continue
    w = (r["weights"] * 100).round(2)
    print(f"--- Weights Used (%): {r['name']} ---")
    print(w[w != 0].sort_values(ascending=False).to_string())
    print()
