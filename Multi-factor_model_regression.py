### Step 1. Import the libraries:
pip install yfinance

import pandas as pd
import yfinance as yf
import statsmodels.formula.api as smf
import pandas_datareader.data as web


### Step 2. Specify the risky asset and the time horizon:
RISKY_ASSET = 'AMZN'
START_DATE = '2000-01-01'
END_DATE = '2020-12-31'


### Step 3. Download the data of the risky asset from Yahoo Finance and Calculate the monthly returns:
asset_df = yf.download(RISKY_ASSET, start=START_DATE, end=END_DATE, progress=False)

y = asset_df['Adj Close'].resample('M').last().pct_change().dropna()
y.index = y.index.strftime('%Y-%m')
y.name = 'return'


### Step 4. Download the risk factors from prof. French's website:
# three factors
df_three_factor = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=START_DATE, end=END_DATE)[0]
df_three_factor.index = df_three_factor.index.format()
# momentum factor
df_mom = web.DataReader('F-F_Momentum_Factor', 'famafrench', start=START_DATE, end=END_DATE)[0]
df_mom.index = df_mom.index.format()



### Step 5. Merge the datasets for the four-factor model:
# join all datasets on the index
four_factor_data = df_three_factor.join(df_mom).join(y)

# rename columns
four_factor_data.columns = ['mkt', 'smb', 'hml', 'rf', 'mom', 'rtn']

# divide everything (except returns) by 100
four_factor_data.loc[:, four_factor_data.columns != 'rtn'] /= 100

# calculate excess returns of risky asset
four_factor_data['excess_rtn'] = four_factor_data.rtn - four_factor_data.rf


### Step 6. Run the regression to estimate alpha and beta
# one-factor model (CAPM):
one_factor_model = smf.ols(formula='excess_rtn ~ mkt', data=four_factor_data).fit()
print(one_factor_model.summary())

# three-factor model:
three_factor_model = smf.ols(formula='excess_rtn ~ mkt + smb + hml', data=four_factor_data).fit()
print(three_factor_model.summary())

# four-factor model:
four_factor_model = smf.ols(formula='excess_rtn ~ mkt + smb + hml + mom', data=four_factor_data).fit()
print(four_factor_model.summary())



#############################
### For five-factor model ###
#############################

### Step 1. Download the risk factors from prof. French's website:
# five factors
df_five_factor = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start=START_DATE, end=END_DATE)[0]
df_five_factor.index = df_five_factor.index.format()

### Step 2. Merge the datasets for the five-factor model:
# join all datasets on the index
five_factor_data = df_five_factor.join(y)

# rename columns
five_factor_data.columns = ['mkt', 'smb', 'hml', 'rmw', 'cma', 'rf', 'rtn']

# divide everything (except returns) by 100
five_factor_data.loc[:, five_factor_data.columns != 'rtn'] /= 100

# calculate excess returns
five_factor_data['excess_rtn'] = five_factor_data.rtn - five_factor_data.rf

### Step 3. Estimate the five-factor model:
five_factor_model = smf.ols(formula='excess_rtn ~ mkt + smb + hml + rmw + cma', data=five_factor_data).fit()
print(five_factor_model.summary())

'''
RMW (Robust Minus Weak) 
Average return on the robust operating profitability portfolios minus the average return on the weak operating profitability portfolios
OP for June of year t is (OP minus interest expense) / book equity for the last fiscal year end in t-1. 
The OP breakpoints are the 30th and 70th NYSE percentiles.

CMA (Conservative Minus Aggressive) 
Average return on the conservative investment portfolios minus the average return on the aggressive investment portfolios
Investment is the change in total assets from the fiscal year ending in year t-2 to the fiscal year ending in t-1, divided by t-2 total assets. 
The Inv breakpoints are the 30th and 70th NYSE percentiles.
'''
