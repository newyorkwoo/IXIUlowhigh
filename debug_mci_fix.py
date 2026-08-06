import yfinance as yf
import pandas as pd
import math

ticker = yf.Ticker('^IXIC')
df = ticker.history(start='2020-01-01')
vix_df = yf.Ticker('^VIX').history(start='2020-01-01')
if hasattr(vix_df.index, 'tz') and vix_df.index.tz is not None: vix_df.index = vix_df.index.tz_localize(None)
if hasattr(df.index, 'tz') and df.index.tz is not None: df.index = df.index.tz_localize(None)
df = df.join(vix_df[['Close']], rsuffix='_VIX')
df.rename(columns={'Close_VIX': 'VIX'}, inplace=True)
df['VIX'] = df['VIX'].ffill().fillna(0)
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain60 = gain.ewm(alpha=1/60, min_periods=60, adjust=False).mean()
avg_loss60 = loss.ewm(alpha=1/60, min_periods=60, adjust=False).mean()
rs60 = avg_gain60 / avg_loss60
df['RSI60'] = 100 - (100 / (1 + rs60))
df['RSI60'] = df['RSI60'].fillna(50)
df.reset_index(inplace=True)
date_col = 'Date' if 'Date' in df.columns else 'Datetime'
df['DateStr'] = df[date_col].dt.strftime('%Y-%m-%d')

df['Daily_MCI'] = 0.0
c_state = 'PEAK'
c_peak_idx = 0
c_peak_val = df['High'].iloc[0]
c_vix_start = df['VIX'].iloc[0]

for i in range(1, len(df)):
    high = df['High'].iloc[i]
    low = df['Low'].iloc[i]
    if c_state == 'PEAK':
        if high > c_peak_val:
            c_peak_val = high
            c_peak_idx = i
            c_vix_start = df['VIX'].iloc[i]
        if low <= c_peak_val * 0.90:
            c_state = 'TROUGH'
    elif c_state == 'TROUGH':
        trough_val = df['Low'].iloc[c_peak_idx:i+1].min()
        if high >= trough_val * 1.10:
            c_state = 'PEAK'
            c_peak_val = high
            c_peak_idx = i
            c_vix_start = df['VIX'].iloc[i]
            
    drop_days = i - c_peak_idx
    drop_pct = (df['Close'].iloc[i] - c_peak_val) / c_peak_val
    if drop_days > 0 and drop_pct < 0:
        price_damage = (abs(drop_pct) * 100) / math.sqrt(drop_days)
        vix_current = df['VIX'].iloc[i]
        fear_premium = (vix_current + max(0, vix_current - c_vix_start)) / 20.0
        rsi_current = df['RSI60'].iloc[i]
        oversold = (50.0 / max(1, rsi_current)) ** 2
        df.at[df.index[i], 'Daily_MCI'] = min(100.0, price_damage * fear_premium * oversold)

mask = (df['DateStr'] >= '2024-07-11') & (df['DateStr'] <= '2024-08-15')
print(df.loc[mask, ['DateStr', 'Close', 'Daily_MCI']].to_string())
