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
symbols = ['GLD', 'SPY', 'MSFT', 'AAPL']
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

# --- 개별 자산 좌표를 symbols 순서로 정렬 ---
idx = [s for s in symbols if s in annual_ret.index]  # 혹시 다운로드 실패 자산 제외
asset_ret = annual_ret.reindex(idx).values
asset_vol = annual_std.reindex(idx).values

# Plot
plt.figure(figsize=(10, 6))
sc = plt.scatter(pvols, prets, c=(prets - risk_free_rate) / pvols,
                 marker='o', cmap='coolwarm', alpha=0.85)
# 개별 자산(노란 점) 별도 표시
plt.scatter(asset_vol, asset_ret, s=70, facecolors='yellow', edgecolors='black', zorder=4)

# 각 자산 라벨
for x, y, name in zip(asset_vol, asset_ret, idx):
    plt.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6),
                 ha='left', fontsize=9, zorder=5)

plt.grid(True, alpha=0.3)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.title('Investment Opportunity Set')
plt.colorbar(sc, label='Sharpe ratio')
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

# --- Plot Efficient Frontier ---
plt.figure(figsize=(10, 6))
plt.scatter(pvols, prets, c=(prets-risk_free_rate) / pvols,
            marker='.', alpha=0.8, cmap='coolwarm')
plt.plot(tvols, trets, 'b', lw=4.0)
plt.plot(port_vol(optv['x']), port_ret(optv['x']),
         'r*', markersize=15.0, label="MVP")
plt.plot(annual_std, annual_ret, 'y.', markersize=15.0)

plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.title("Efficient Frontier")   
plt.legend()
plt.show()

# ================== Pretty MVP Output ==================
mvp_w     = optv.x
mvp_vol   = float(port_vol(mvp_w))
mvp_ret   = float(port_ret(mvp_w))
mvp_sr    = float((mvp_ret - risk_free_rate) / mvp_vol)

# Weights as % only
weights_s = pd.Series(mvp_w, index=symbols)
weights_s[weights_s.abs() < 1e-6] = 0.0  # 작은 값은 0 처리
weights_df = pd.DataFrame({
    "Weight (%)": (weights_s * 100).round(2)   
}).sort_values("Weight (%)", ascending=False)

print("\n================ Minimum-Variance Portfolio (Long-only) ================")
print("Assets & Weights (%):\n")
print(weights_df.to_string())
print("\nSummary Stats:")
print(f"- Volatility (σ):  {mvp_vol:.3f}")
print(f"- Return (μ):      {mvp_ret:.3f}")
print(f"- Sharpe Ratio:    {mvp_sr:.3f}   (rf = {risk_free_rate:.2%})")
print("=======================================================================\n")

# (선택) 가중치 바 차트
plt.figure(figsize=(7,4))
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

# ORP weights (%만 표시)
orp_weights_s = pd.Series(orp_w, index=symbols)
orp_weights_s[orp_weights_s.abs() < 1e-6] = 0.0  # tiny noise -> 0
orp_weights_df = pd.DataFrame({
    "Weight (%)": (orp_weights_s * 100).round(2)
}).sort_values("Weight (%)", ascending=False)

# ---------- CML 준비 (효율적 경계 스플라인) ----------
# (Step 1-2에서 계산된 tvols, trets, max_return 사용)
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
plt.figure(figsize=(10, 7))
sc = plt.scatter(pvols, prets, c=(prets - risk_free_rate) / pvols,
                 marker='.', cmap='coolwarm', alpha=0.85, label='Random Portfolios')
plt.plot(evols, erets, 'b', lw=4.0, label='Efficient Frontier')

# CML
cx = np.linspace(0.0, max_return + 0.05, 200)
plt.plot(cx, opt[0] + opt[1]*cx, 'r', lw=1.8, label='Capital Market Line')

# Tangency point & ORP
plt.plot(opt[2], f(opt[2]), 'y*', markersize=15.0, label='Tangency Point')
plt.plot(orp_vol, orp_ret, 'ks', markersize=6, label='ORP (Max Sharpe)')

# 개별 자산
plt.plot(annual_std, annual_ret, 'c.', markersize=15.0, label='Assets')

plt.grid(True, alpha=0.3)
plt.axhline(0, color='k', ls='--', lw=1.2)
plt.axvline(0, color='k', ls='--', lw=1.2)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(sc, label='Sharpe ratio')
plt.title("Optimal Risky Portfolio & CML")   
plt.legend(loc='best')
plt.tight_layout()

# ---------- 그래프 아래에 ORP 결과값 표시 ----------
result_text = (
    "================ Optimal Risk Portfolio (Max Sharpe, Long-only) ================\n"
    "Assets & Weights (%):\n"
    f"{orp_weights_df.to_string()}\n\n"
    "Summary Stats:\n"
    f"- Volatility (σ):  {orp_vol:.3f}\n"
    f"- Return (μ):      {orp_ret:.3f}\n"
    f"- Sharpe Ratio:    {orp_sr:.3f}   (rf = {risk_free_rate:.2%})\n"
    "===============================================================================\n"
)

# y 좌표는 그림 아래로 살짝 내림 (필요시 -0.25 ~ -0.35 사이 조절)
plt.figtext(0.5, -0.27, result_text, wrap=True, ha="center", va="top", fontsize=9, family='monospace')

plt.show()

# (선택) ORP 가중치 바 차트
plt.figure(figsize=(7,4))
(orp_weights_s * 100).sort_values(ascending=False).plot(kind='bar')
plt.title("ORP Weights by Asset")
plt.ylabel("Weight (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
