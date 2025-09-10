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

# 3) Params
symbols = ['SPY', 'TLT', 'IEF', 'GLD']
start_date = '2010-01-04'
end_date   = '2025-08-31'
risk_free_rate = 0.02

# You can construct your preferred multi-asset portfolio
# symbols = ['SPY', 'TLT', 'IEF', 'GLD']
# start_date = '2004-11-18' 
# start date is GLD inception date


# 4) Download prices (ensure Adj Close is present)
data = yf.download(symbols, start=start_date, end=end_date, auto_adjust=False)['Adj Close']

# 5) Charts
(data / data.iloc[0] * 100).plot(figsize=(8,5), title='Price (Indexed to 100)')
plt.ylabel('Index Level'); plt.show()

rets = np.log(data / data.shift(1)).dropna()
rets.hist(bins=40, figsize=(8,5))
plt.suptitle('Daily Log Return Distributions'); plt.show()

# 6) Annualized stats
annual_ret = rets.mean() * 252
annual_std = rets.std() * math.sqrt(252)
annual_cov = rets.cov() * 252

# Combine into a table and print return, SD, Covariance
summary_table = pd.DataFrame({ 'R': annual_ret.round(4), 'SD': annual_std.round(4) })

print("\n=== Annualized Returns and Standard Deviations ===")
print(summary_table)

print("\n=== Annualized Covariance Matrix ===")
print(annual_cov.round(4))


#########################################################
### Step 1-1                                           ##   
### Investment Opportunity Sets (labels fixed)         ##
#########################################################

def port_ret(weights):
    return weights.T @ annual_ret  # from Step 0

def port_vol(weights):
    return (weights.T @ annual_cov @ weights) ** 0.5  # from Step 0

noa = len(symbols)

# Monte Carlo simulation
prets, pvols = [], []
for _ in range(2500):
    w = np.random.random(noa)
    w /= w.sum()
    prets.append(port_ret(w))
    pvols.append(port_vol(w))
prets = np.array(prets)
pvols = np.array(pvols)

# --- Arrange individual asset coordinates in symbols order ---
idx = [s for s in symbols if s in annual_ret.index]  # Exclude assets that failed download
asset_ret = annual_ret.reindex(idx).values
asset_vol = annual_std.reindex(idx).values

# Plot
plt.figure(figsize=(8, 5))
sc = plt.scatter(pvols, prets, c=(prets - risk_free_rate) / pvols,
                 marker='o', cmap='coolwarm', alpha=0.85)
# Highlight individual assets (yellow dots)
plt.scatter(asset_vol, asset_ret, s=70, facecolors='yellow', edgecolors='black', zorder=4)

# Labels for each asset
for x, y, name in zip(asset_vol, asset_ret, idx):
    plt.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6),
                 ha='left', fontsize=9, zorder=5)

plt.grid(True, alpha=0.3)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.title('Investment Opportunity Set')
plt.show()

#########################################################
### Step 1-2                                           ##   
### Efficient Frontier                                 ##
#########################################################

import scipy.optimize as sco

# Minimum Variance Portfolio (MVP)
bnds = tuple((0, 1) for x in range(noa))
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
eweights = np.array(noa * [1. / noa,])
optv = sco.minimize(port_vol, eweights, method='SLSQP',
                    bounds=bnds, constraints=cons)

# Efficient Frontier => Find min vol at each target return
cons = ({'type': 'eq', 'fun': lambda x: port_ret(x) - tret},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))

min_return = port_ret(optv['x'])
max_return = np.max(annual_ret)

trets = np.linspace(min_return, max_return, 50)
tvols = []
for tret in trets:
    res = sco.minimize(port_vol, eweights, method='SLSQP',
                       bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)

# Compute MVP stats
mvp_w   = optv.x
mvp_vol = float(port_vol(mvp_w))
mvp_ret = float(port_ret(mvp_w))
mvp_sr  = float((mvp_ret - risk_free_rate) / mvp_vol)

# Weights as % only
weights_s = pd.Series(mvp_w, index=symbols)
weights_s[weights_s.abs() < 1e-6] = 0.0  # small values → 0
weights_df = pd.DataFrame({
    "Weight (%)": (weights_s * 100).round(2)   
}).sort_values("Weight (%)", ascending=False)

# --- Efficient Frontier + MVP text block (same style as ORP) ---
fig = plt.figure(figsize=(7, 8), constrained_layout=True)
gs  = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[6, 1], hspace=0.03)

ax = fig.add_subplot(gs[0])      # Top: graph
ax_txt = fig.add_subplot(gs[1])  # Bottom: text
ax_txt.axis('off')

sc = ax.scatter(pvols, prets, c=(prets - risk_free_rate) / pvols,
                marker='o', alpha=0.85, cmap='coolwarm', label='Random Portfolios')
ax.plot(tvols, trets, 'b', lw=4.0, label='Efficient Frontier')
ax.plot(port_vol(optv['x']), port_ret(optv['x']),
        'ks', markersize=8.0, label="MVP")  # black square marker

# Highlight individual assets (yellow dots with black outline) + labels
ax.scatter(annual_std, annual_ret, s=70, facecolors='yellow',
           edgecolors='black', zorder=4, label='Assets')
for x, y, name in zip(annual_std, annual_ret, annual_ret.index):
    ax.annotate(name, (x, y), textcoords="offset points",
                xytext=(6, 6), ha='left', fontsize=9, zorder=5)

ax.grid(True, alpha=0.3)
ax.set_xlabel('expected volatility')
ax.set_ylabel('expected return')
ax.set_title("Efficient Frontier")
ax.legend(loc='best')

# MVP text block under the graph
result_text = (
    "\n"  # blank line above header
    "\n"  # blank line above header
    "================ Minimum-Variance Portfolio (Long-only) ================\n"
    "\n"  # blank line after header
    "Assets & Weights (%):\n"
    f"\n{weights_df.to_string()}\n\n"   # extra blank lines around table
    "Summary Stats:\n"
    f"- Volatility (σ):  {mvp_vol:.3f}\n"
    f"- Return (μ):      {mvp_ret:.3f}\n"
    f"- Sharpe Ratio:    {mvp_sr:.3f}  \n"
    "\n"  # blank line before footer
    "=======================================================================\n"
    )
ax_txt.text(0.5, 1.0, result_text, ha='center', va='top',
            fontsize=9, family='monospace', transform=ax_txt.transAxes)

plt.show()

# --- MVP Weights Bar Chart (40% smaller) ---
plt.figure(figsize=(4.2, 2.4))
(weights_s * 100).sort_values(ascending=False).plot(kind='bar')
plt.title("MVP Weights by Asset")
plt.ylabel("Weight (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#########################################################
### Step 2                                             ##   
### Optimal Risk Portfolio and Capital Market Line     ##
#########################################################

import scipy.optimize as sco
import scipy.interpolate as sci

# ---------- ORP (Max Sharpe) ----------
def min_func_sharpe(weights):
    return -(port_ret(weights) - risk_free_rate) / port_vol(weights)

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
orp_weights_s = pd.Series(orp_w, index=symbols)
orp_weights_s[orp_weights_s.abs() < 1e-6] = 0.0
orp_weights_df = pd.DataFrame({
    "Weight (%)": (orp_weights_s * 100).round(2)
}).sort_values("Weight (%)", ascending=False)

# ---------- CML prep (efficient frontier spline) ----------
ind  = np.argmin(tvols)
evols = tvols[ind:]
erets = trets[ind:]
tck = sci.splrep(evols, erets)

def f(x):   # frontier
    return sci.splev(x, tck, der=0)

def df(x):  # slope
    return sci.splev(x, tck, der=1)

def equations(p, rf=risk_free_rate):
    # p = [a, b, x*]  (a=rf, b=slope, x*=tangent volatility)
    eq1 = rf - p[0]
    eq2 = rf + p[1]*p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3

a0 = risk_free_rate
b0 = orp_sr
x0 = port_vol(opts['x'])
opt = sco.fsolve(equations, [a0, b0, x0])  # [a, b, x*]

# ---------- Plot: ORP & CML ----------
fig = plt.figure(figsize=(7, 8), constrained_layout=True)
gs  = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[6, 1], hspace=0.03)

ax = fig.add_subplot(gs[0])        # Top: graph
ax_txt = fig.add_subplot(gs[1])    # Bottom: text
ax_txt.axis('off')

sc = ax.scatter(pvols, prets, c=(prets - risk_free_rate) / pvols,
                marker='.', cmap='coolwarm', alpha=0.85, label='Random Portfolios')
ax.plot(evols, erets, 'b', lw=4.0, label='Efficient Frontier')

# CML
cx = np.linspace(0.0, max_return + 0.05, 200)
ax.plot(cx, opt[0] + opt[1]*cx, 'r', lw=1.8, label='Capital Market Line')

# ORP (Yellow Star with Black Outline)
ax.scatter(orp_vol, orp_ret, s=250, marker='*',
           facecolors='yellow', edgecolors='black',
           linewidths=1.2, zorder=5, label='ORP (Max Sharpe)')

# Individual assets (yellow dots + labels)
ax.scatter(annual_std, annual_ret, s=70, facecolors='yellow', edgecolors='black', zorder=4, label='Assets')
for x, y, name in zip(annual_std, annual_ret, annual_ret.index):
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6),
                ha='left', fontsize=9, zorder=5)

ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', ls='--', lw=1.2)
ax.axvline(0, color='k', ls='--', lw=1.2)
ax.set_xlabel('expected volatility')
ax.set_ylabel('expected return')
ax.set_title("Optimal Risky Portfolio & CML")
ax.legend(loc='best')


# Text right below the chart (keep monospace alignment)
result_text = (
    "\n"  # blank line above header
    "\n"  # blank line above header
    "================ Optimal Risk Portfolio (Max Sharpe, Long-only) ================\n"
    "\n"  # blank line after header
    "Assets & Weights (%):\n"
    f"{orp_weights_df.to_string()}\n\n"
    "Summary Stats:\n"
    f"- Volatility (σ):  {orp_vol:.3f}\n"
    f"- Return (μ):      {orp_ret:.3f}\n"
    f"- Sharpe Ratio:    {orp_sr:.3f}\n"
    f"  (rf = {risk_free_rate:.2%})\n"
    "\n"  # blank line before footer
    "===============================================================================\n"
)
ax_txt.text(0.5, 1.0, result_text, ha='center', va='top',
            fontsize=9, family='monospace', transform=ax_txt.transAxes)

plt.show()

# ORP weights bar chart (40% smaller)
plt.figure(figsize=(4.2, 2.4))
(orp_weights_s * 100).sort_values(ascending=False).plot(kind='bar')
plt.title("ORP Weights by Asset")
plt.ylabel("Weight (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


#########################################################
### Step 0                                             ##   
### Select your assets, data range and rf and SR       ##
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

# 3) Params
symbols = ['SPY', 'EWU', 'EWQ', 'EWG']
start_date = '2010-01-01'
end_date   = '2025-08-31'
risk_free_rate = 0.02   # assumed annual risk-free rate

# 4) Download prices
data = yf.download(symbols, start=start_date, end=end_date, auto_adjust=False)['Adj Close']

# 5) Charts
(data / data.iloc[0] * 100).plot(figsize=(8,5), title='Price (Indexed to 100)')
plt.ylabel('Index Level'); plt.show()

rets = np.log(data / data.shift(1)).dropna()
rets.hist(bins=40, figsize=(8,5))
plt.suptitle('Daily Log Return Distributions'); plt.show()

# 6) Annualized stats
annual_ret = rets.mean() * 252
annual_std = rets.std() * math.sqrt(252)
annual_cov = rets.cov() * 252
annual_corr = rets.corr()

# Sharpe Ratio
sharpe_ratio = (annual_ret - risk_free_rate) / annual_std

# Summary table
summary_table = pd.DataFrame({
    'R': annual_ret.round(4),
    'SD': annual_std.round(4),
    'SR': sharpe_ratio.round(4)
})

print("\n=== Annualized Returns, Standard Deviations, and Sharpe Ratios ===")
print(summary_table)

print("\n=== Annualized Covariance Matrix ===")
print(annual_cov.round(4))

print("\n=== Correlation Matrix ===")
print(annual_corr.round(4))
