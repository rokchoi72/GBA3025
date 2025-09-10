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


#########################################################
### Step 1-1                                           ##   
### Investment Opportunity Sets                        ##
#########################################################


def port_ret(weights):
    return weights.T @ annual_ret

def port_vol(weights):
    return (weights.T @ annual_cov @ weights )**0.5

noa = len(symbols) # noa = number of assets

prets = []
pvols = []
for p in range (2500):  #Assume the number of simulation (here 2500 but better to have > 100,000)
    weights = np.random.random(noa) 
    weights /= np.sum(weights)
    prets.append(port_ret(weights))
    pvols.append(port_vol(weights))
prets = np.array(prets)
pvols = np.array(pvols)

plt.figure(figsize=(10, 6))
plt.scatter(pvols, prets, c=(prets-risk_free_rate) / pvols, marker='o', cmap='coolwarm')
plt.plot(annual_std, annual_ret, 'y.', markersize=15.0)
plt.grid()
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio');

# Combine into a table and print return, SD, Correlation
summary_table = pd.DataFrame({ 'R': annual_ret, 'SD': annual_std })

print("Annualized Returns and Standard Deviations:")
print(summary_table)

print("\nAnnualized Covariance Matrix:")
print(annual_cov)

# print(f'Volatiltiy: \n{(annual_std).round(4)} \n\nReturn: \n{(annual_ret).round(4)} \n\nCorrelation: \n{rets.corr().round(4)}')


#########################################################
### Step 1-2                                             ##   
### Efficient Frontier                                 ##
#########################################################

import scipy.optimize as sco


# Minium Variance Portfolio
bnds = tuple((0, 1) for x in range(noa))
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
eweights = np.array(noa * [1. / noa,])
optv = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)


# Efficient Frontier => Find min vol at each target return
cons = ({'type': 'eq', 'fun': lambda x: port_ret(x) - tret},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))

min_return = port_ret(optv['x'])
max_return = np.max(annual_ret)

trets = np.linspace(min_return, max_return, 50)
tvols = []
for tret in trets:
    res = sco.minimize(port_vol, eweights, method='SLSQP', bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)

plt.figure(figsize=(10, 6))
plt.scatter(pvols, prets, c=(prets-risk_free_rate) / pvols, marker='.', alpha=0.8, cmap='coolwarm')
plt.plot(tvols, trets, 'b', lw=4.0)
plt.plot(port_vol(optv['x']), port_ret(optv['x']), 'r*', markersize=15.0)
plt.plot(annual_std, annual_ret, 'y.', markersize=15.0)
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio');

# ================== Pretty MVP Output ==================
# MVP metrics
mvp_w     = optv.x
mvp_vol   = float(port_vol(mvp_w))
mvp_ret   = float(port_ret(mvp_w))
mvp_sr    = float((mvp_ret - risk_free_rate) / mvp_vol)

# Weights as a labeled Series (by asset name)
weights_s = pd.Series(mvp_w, index=symbols)
weights_s[weights_s.abs() < 1e-6] = 0.0  # tiny noise -> 0
weights_df = pd.DataFrame({
    "Weight": weights_s.round(4),
    "Weight (%)": (weights_s * 100).round(2)
}).sort_values("Weight", ascending=False)

print("\n================ Minimum-Variance Portfolio (Long-only) ================")
print("Assets & Weights:\n")
print(weights_df.to_string())
print("\nSummary Stats:")
print(f"- Volatility (σ):  {mvp_vol:.3f}")
print(f"- Return (μ):      {mvp_ret:.3f}")
print(f"- Sharpe Ratio:    {mvp_sr:.3f}   (rf = {risk_free_rate:.2%})")
print("=======================================================================\n")

# (선택) 효율적 경계 그래프에 MVP 포인트 표시
# plt.plot(mvp_vol, mvp_ret, marker='*', color='red', markersize=15)
# plt.annotate("MVP", xy=(mvp_vol, mvp_ret), xytext=(mvp_vol*1.02, mvp_ret*0.98),
#              arrowprops=dict(arrowstyle='->', lw=1))

# (선택) 가중치 바 차트
plt.figure(figsize=(7,4))
(weights_s * 100).sort_values(ascending=False).plot(kind='bar')
plt.title("MVP Weights by Asset")
plt.ylabel("Weight (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
# =======================================================

#########################################################
### Step 2                                             ##   
### Optimal Risk Portfolio and Capital Market Line     ##
#########################################################

# Optimal Risk Portfolio (ORP)
def min_func_sharpe(weights):
    return -(port_ret(weights)-risk_free_rate) / port_vol(weights)

bnds = tuple((0, 1) for x in range(noa))
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
eweights = np.array(noa * [1. / noa,])
opts = sco.minimize(min_func_sharpe, eweights, method='SLSQP', bounds=bnds, constraints=cons)

# ============= ORP(최적 위험 포트폴리오) 가중치 예쁘게 출력 =============
orp_w   = opts.x
orp_vol = float(port_vol(orp_w))
orp_ret = float(port_ret(orp_w))
orp_sr  = float((orp_ret - risk_free_rate) / orp_vol)

# 자산명으로 라벨링
orp_weights_s = pd.Series(orp_w, index=symbols)
orp_weights_s[orp_weights_s.abs() < 1e-6] = 0.0  # 노이즈 제거

orp_weights_df = pd.DataFrame({
    "Weight": orp_weights_s.round(4),
    "Weight (%)": (orp_weights_s * 100).round(2)
}).sort_values("Weight", ascending=False)

print("\n================ Optimal Risk Portfolio (Max Sharpe, Long-only) ================")
print("Assets & Weights:\n")
print(orp_weights_df.to_string())
print("\nSummary Stats:")
print(f"- Volatility (σ):  {orp_vol:.3f}")
print(f"- Return (μ):      {orp_ret:.3f}")
print(f"- Sharpe Ratio:    {orp_sr:.3f}   (rf = {risk_free_rate:.2%})")
print("===============================================================================\n")

# (선택) ORP 가중치 바 차트
plt.figure(figsize=(7,4))
(orp_weights_s * 100).sort_values(ascending=False).plot(kind='bar')
plt.title("ORP Weights by Asset")
plt.ylabel("Weight (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# (참고) CML 식도 출력 (여기서만 1회 출력)
print(f"Capital Market Line:  Rp = {risk_free_rate:.3f} + {orp_sr:.3f} * Volatility")


# ===================== Capital Market Line (CML) Plot ===================== 
import scipy.interpolate as sci

# 효율적 경계에서 최소 변동성 이후 구간만 스플라인
ind = np.argmin(tvols)
evols = tvols[ind:]
erets = trets[ind:]

tck = sci.splrep(evols, erets)

def f(x):
    """Efficient frontier function (spline approximation)."""
    return sci.splev(x, tck, der=0)

def df(x):
    """First derivative of efficient frontier function."""
    return sci.splev(x, tck, der=1)

def equations(p, rf=risk_free_rate):
    # p = [a, b, x*] where a=rf, b=slope, x*=tangent point volatility
    eq1 = rf - p[0]                # a = rf
    eq2 = rf + p[1] * p[2] - f(p[2])  # line meets frontier at x*
    eq3 = p[1] - df(p[2])          # slope matches derivative at x*
    return eq1, eq2, eq3

a0 = risk_free_rate
b0 = orp_sr
x0 = port_vol(opts['x'])  # 초기값은 탄젠트 포트폴리오 변동성으로 설정 (중요!)
opt = sco.fsolve(equations, [a0, b0, x0])  # opt = [a, b, x*]

plt.figure(figsize=(10, 6))
plt.scatter(pvols, prets, c=(prets - risk_free_rate) / pvols, marker='.', cmap='coolwarm')
plt.plot(evols, erets, 'b', lw=4.0, label='Efficient Frontier')

# CML 라인
cx = np.linspace(0.0, max_return + 0.05)
plt.plot(cx, opt[0] + opt[1] * cx, 'r', lw=1.5, label='Capital Market Line')

# 탄젠트 포인트 & ORP 표시
plt.plot(opt[2], f(opt[2]), 'y*', markersize=15.0, label='Tangency Point')
plt.plot(orp_vol, orp_ret, 'ks', markersize=6, label='ORP (Max Sharpe)')

# 개별 자산 점
plt.plot(annual_std, annual_ret, 'c.', markersize=15.0, label='Assets')

plt.grid(True)
plt.axhline(0, color='k', ls='--', lw=2.0)
plt.axvline(0, color='k', ls='--', lw=2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
