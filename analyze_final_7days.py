import yfinance as yf
import pandas as pd
import numpy as np

# Load Data
df = yf.Ticker('^IXIC').history(start='2000-01-01')
if hasattr(df.index, 'tz') and df.index.tz is not None:
    df.index = df.index.tz_localize(None)

df['ROC7'] = df['Close'].pct_change(periods=7) * 100
df.reset_index(inplace=True)
date_col = 'Date' if 'Date' in df.columns else 'Datetime'
df['DateStr'] = df[date_col].dt.strftime('%Y-%m-%d')

periods = []
state = 'PEAK'
peak_idx = 0
peak_val = df['High'].iloc[0]

trough_idx = 0
trough_val = df['Low'].iloc[0]

for i in range(1, len(df)):
    high = df['High'].iloc[i]
    low = df['Low'].iloc[i]
    
    if state == 'PEAK':
        if high > peak_val:
            peak_val = high
            peak_idx = i
        if low <= peak_val * 0.90:
            state = 'TROUGH'
            trough_val = low
            trough_idx = i
    elif state == 'TROUGH':
        if low < trough_val:
            trough_val = low
            trough_idx = i
        if high >= trough_val * 1.10:
            # We have a confirmed wave. Trough is at trough_idx.
            final_7d_speed = df['ROC7'].iloc[trough_idx]
            wave_total_drop = (trough_val - peak_val) / peak_val * 100
            
            periods.append({
                'Trough Date': df['DateStr'].iloc[trough_idx],
                'Total Drop': wave_total_drop,
                'Final 7d Speed': final_7d_speed
            })
            
            state = 'PEAK'
            peak_val = high
            peak_idx = i

# Put into DF and analyze
res = pd.DataFrame(periods)
# Drop na just in case
res = res.dropna()

print(f"Total >10% Bottoms Analyzed: {len(res)}")
print(f"Average Final 7-Day Drop: {res['Final 7d Speed'].mean():.2f}%")
print(f"Median Final 7-Day Drop: {res['Final 7d Speed'].median():.2f}%")

print("\n--- Distribution of Final 7-Day Drop ---")
print(f"More than -10%: {len(res[res['Final 7d Speed'] <= -10])} times")
print(f"-5% to -10%: {len(res[(res['Final 7d Speed'] > -10) & (res['Final 7d Speed'] <= -5)])} times")
print(f"0% to -5%: {len(res[(res['Final 7d Speed'] > -5) & (res['Final 7d Speed'] <= 0)])} times")
print(f"Positive (> 0%): {len(res[res['Final 7d Speed'] > 0])} times") # meaning the absolute lowest low happened, but the *close* 7 days prior was actually lower, or the drop was completely flat/V-shape intra-week.

print("\n--- Notable Fast Final 7 Days ---")
print(res.sort_values('Final 7d Speed').head(10).to_string(index=False))

print("\n--- Summary Stats ---")
print(res['Final 7d Speed'].describe())
