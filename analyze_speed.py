import yfinance as yf
import pandas as pd

df = yf.Ticker('^IXIC').history(start='2000-01-01')
vix_df = yf.Ticker('^VIX').history(start='2000-01-01')

if hasattr(vix_df.index, 'tz') and vix_df.index.tz is not None:
    vix_df.index = vix_df.index.tz_localize(None)
if hasattr(df.index, 'tz') and df.index.tz is not None:
    df.index = df.index.tz_localize(None)

df = df.join(vix_df[['Close']], rsuffix='_VIX')
df.rename(columns={'Close_VIX': 'VIX'}, inplace=True)

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.ewm(alpha=1/60, min_periods=60, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/60, min_periods=60, adjust=False).mean()
rs = avg_gain / avg_loss
df['RSI60'] = 100 - (100 / (1 + rs))

df['VIX'] = df['VIX'].fillna(0)
df['RSI60'] = df['RSI60'].fillna(50)

df.reset_index(inplace=True)
date_col = 'Date' if 'Date' in df.columns else 'Datetime'
df['DateStr'] = df[date_col].dt.strftime('%Y-%m-%d')

drawdown_periods = []
state = 'PEAK'
peak_idx = 0
peak_val = df['High'].iloc[0]

trough_idx = 0
trough_val = df['Low'].iloc[0]

drop_threshold = 0.10
recovery_threshold = 0.10

for i in range(1, len(df)):
    high = df['High'].iloc[i]
    low = df['Low'].iloc[i]
    
    if state == 'PEAK':
        if high > peak_val:
            peak_val = high
            peak_idx = i
        if low <= peak_val * (1 - drop_threshold):
            state = 'TROUGH'
            trough_val = low
            trough_idx = i
    elif state == 'TROUGH':
        if low < trough_val:
            trough_val = low
            trough_idx = i
        if high >= trough_val * (1 + recovery_threshold):
            drop_days = trough_idx - peak_idx
            drop_pct = (trough_val - peak_val) / peak_val
            
            period_df = df.iloc[peak_idx:trough_idx+1]
            max_vix = period_df['VIX'].max()
            min_rsi = period_df['RSI60'].min()
            
            drawdown_periods.append({
                'start': df['DateStr'].iloc[peak_idx],
                'end': df['DateStr'].iloc[trough_idx],
                'drop_pct': drop_pct,
                'drop_days': drop_days,
                'drop_speed': (drop_pct / drop_days) * 100 if drop_days > 0 else drop_pct * 100,
                'max_vix': max_vix,
                'min_rsi': min_rsi
            })
            state = 'PEAK'
            peak_val = high
            peak_idx = i

bottoms = [p for p in drawdown_periods if p['max_vix'] > 28 and p['min_rsi'] <= 42]
bottoms.sort(key=lambda x: x['drop_speed'])

for b in bottoms:
    print(f"Date: {b['start']} to {b['end']} | Drop: {b['drop_pct']*100:.2f}% | Days: {b['drop_days']:3d} | Speed: {b['drop_speed']:.2f}%/day | VIX: {b['max_vix']:.2f} | RSI: {b['min_rsi']:.2f}")

speeds = [b['drop_speed'] for b in bottoms]
print(f"\nStats for Bottom Drop Speeds:")
print(f"Average: {sum(speeds)/len(speeds):.2f}%/day")
print(f"Min (Fastest): {min(speeds):.2f}%/day")
print(f"Max (Slowest): {max(speeds):.2f}%/day")
