## Import libraries

import pandas as pd
import yfinance as yf
import pandas_datareader
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import seaborn as sns


## Step 1. Define the data period
START_DATE = '1999-01-01'
END_DATE = '2023-8-31'


### Step 2. Download the risk factors from prof. French's website:
df_five_factor = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start=START_DATE, end=END_DATE)[0]
df_mom = web.DataReader('F-F_Momentum_Factor', 'famafrench', start=START_DATE, end=END_DATE)[0]


## Step 3. Merges the two downloaded datasets (df_five_factor and df_mom)
result = pd.merge(df_five_factor, df_mom, left_index=True, right_index=True)


# Step 4. Graph for cumulative returns of factor returns
((result/100 + 1).cumprod() - 1).plot(figsize=(10,6))
# plt.legend(loc=2)


# Step 5. Calculate correlation between factors
correlation_matrix = result.drop(['RF'], axis=1).corr().round(4)
plt.figure(figsize=(8, 4))
sns.set(font_scale=1.2)  # Adjust the font size for better readability
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


## Step 6. If I add two reversal factors
df_str = web.DataReader('F-F_ST_Reversal_Factor', 'famafrench', start=START_DATE, end=END_DATE)[0]
df_ltr = web.DataReader('F-F_LT_Reversal_Factor', 'famafrench', start=START_DATE, end=END_DATE)[0]
result1 = pd.merge(df_five_factor, df_mom, left_index=True, right_index=True)
result2 = pd.merge(df_str, df_ltr, left_index=True, right_index=True)
result = pd.merge(result1, result2, left_index=True, right_index=True)
((result/100 + 1).cumprod() - 1).plot(figsize=(10,6))

correlation_matrix = result.drop(['RF'], axis=1).corr().round(4)
plt.figure(figsize=(8, 4))
sns.set(font_scale=1.2)  # Adjust the font size for better readability
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


## You can check the available data from french library and add to your correlation calculation 
pandas_datareader.famafrench.get_available_datasets()
