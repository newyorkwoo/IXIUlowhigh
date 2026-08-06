import yfinance as yf
import pandas as pd

df = yf.Ticker('^IXIC').history(start='2023-07-01', end='2023-11-10')
vix_df = yf.Ticker('^VIX').history(start='2023-07-01', end='2023-11-10')

if hasattr(vix_df.index, 'tz') and vix_df.index.tz is not None:
    vix_df.index = vix_df.index.tz_localize(None)
if hasattr(df.index, 'tz') and df.index.tz is not None:
    df.index = df.index.tz_localize(None)

df = df.join(vix_df[['Close']], rsuffix='_VIX')
df.rename(columns={'Close_VIX': 'VIX'}, inplace=True)

df['MA60'] = df['Close'].rolling(60).mean().bfill()
df['MA240'] = df['Close'].rolling(240).mean().bfill()

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain60 = gain.ewm(alpha=1/60, min_periods=60, adjust=False).mean()
avg_loss60 = loss.ewm(alpha=1/60, min_periods=60, adjust=False).mean()
df['RSI60'] = 100 - (100 / (1 + avg_gain60 / avg_loss60))

avg_gain14 = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
avg_loss14 = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
df['RSI14'] = 100 - (100 / (1 + avg_gain14 / avg_loss14))

# For MA240, we need more history. Let's fetch more history just for MA calculation
df_full = yf.Ticker('^IXIC').history(start='2022-01-01', end='2023-11-10')
if hasattr(df_full.index, 'tz') and df_full.index.tz is not None:
    df_full.index = df_full.index.tz_localize(None)
df_full['MA60'] = df_full['Close'].rolling(60).mean()
df_full['MA240'] = df_full['Close'].rolling(240).mean()

# Merge MA240 back
df['MA240'] = df_full['MA240']
df['MA60'] = df_full['MA60']

print(df.loc['2023-10-20':'2023-11-01', ['Close', 'Low', 'MA60', 'MA240', 'VIX', 'RSI14', 'RSI60']])

