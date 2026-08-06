import yfinance as yf
import pandas as pd

df = yf.Ticker('^IXIC').history(start='2022-01-01', end='2023-11-10')
vix_df = yf.Ticker('^VIX').history(start='2022-01-01', end='2023-11-10')

if hasattr(vix_df.index, 'tz') and vix_df.index.tz is not None:
    vix_df.index = vix_df.index.tz_localize(None)
if hasattr(df.index, 'tz') and df.index.tz is not None:
    df.index = df.index.tz_localize(None)

df = df.join(vix_df[['Close']], rsuffix='_VIX')
df.rename(columns={'Close_VIX': 'VIX'}, inplace=True)

df['MA240'] = df['Close'].rolling(240).mean()

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain60 = gain.ewm(alpha=1/60, min_periods=60, adjust=False).mean()
avg_loss60 = loss.ewm(alpha=1/60, min_periods=60, adjust=False).mean()
df['RSI60'] = 100 - (100 / (1 + avg_gain60 / avg_loss60))

# Check 2023-10-26
print("2023-10-26 stats:")
print(df.loc['2023-10-26', ['Close', 'Low', 'MA240', 'VIX', 'RSI60']])

print("\nDistance to MA240:")
print(df.loc['2023-10-26', 'Low'] / df.loc['2023-10-26', 'MA240'])

