pip install zipline-reloaded
pip install pyfolio

###1. Import the libraries:
import yfinance as yf
import numpy as np
import pandas as pd
import pyfolio as pf


###2. Set up the parameters:
RISKY_ASSETS = ['SPY', 'TLT', 'IEF', 'GLD']
START_DATE = '2005-01-01'
END_DATE = '2020-12-31'
n_assets = len(RISKY_ASSETS)


###3. Download the stock prices from Yahoo Finance:
prices_df = yf.download(RISKY_ASSETS, start=START_DATE, end=END_DATE)


###4. Calculate individual asset returns:
returns = prices_df['Adj Close'].pct_change().dropna()


###5. Define the weights:
portfolio_weights = n_assets * [1 / n_assets]


###6. Calculate the portfolio returns:
portfolio_returns = pd.Series(np.dot(portfolio_weights, returns.T), index=returns.index)


###7. Create the tear sheet (simple variant):
pf.create_simple_tear_sheet(portfolio_returns)
