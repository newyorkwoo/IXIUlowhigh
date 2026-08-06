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

df['VIX'] = df['VIX'].fillna(0)
df['RSI60'] = df['RSI60'].fillna(50)
df['RSI14'] = df['RSI14'].fillna(50)

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
            drop_days = trough_idx - peak_idx
            drop_pct = (trough_val - peak_val) / peak_val
            
            # evaluate signals within a few days around trough
            t_start = max(0, trough_idx - 5)
            t_end = min(len(df)-1, trough_idx + 5)
            period_df = df.iloc[t_start:t_end+1]
            
            # max/min near trough
            vix_max = period_df['VIX'].max()
            rsi60_min = period_df['RSI60'].min()
            rsi14_min = period_df['RSI14'].min()
            
            # Check conditions
            trough_row = df.iloc[trough_idx]
            isGrind = trough_row['Low'] <= trough_row['MA240'] * 1.05 and rsi60_min <= 46 and vix_max <= 28
            isStructural = vix_max > 28 and rsi60_min <= 42
            isFlash = vix_max > 35 and rsi14_min <= 30
            
            label = []
            if isStructural: label.append('🟡 長線恐慌底')
            if isFlash: label.append('🔵 閃崩底')
            if isGrind and not isStructural: label.append('🟢 陰跌支撐底')
            
            if not label: label = ['⚪ 無極端訊號']
            
            periods.append({
                'Period': f"{df['DateStr'].iloc[peak_idx]} ~ {df['DateStr'].iloc[trough_idx]}",
                'Drop': f"{drop_pct*100:.2f}%",
                'Days': drop_days,
                'Speed': f"{(drop_pct/drop_days)*100 if drop_days > 0 else drop_pct*100:.2f}%/天",
                'VIX_Peak': f"{vix_max:.2f}",
                'RSI60_Low': f"{rsi60_min:.2f}",
                'Signal': ' + '.join(label)
            })
            state = 'PEAK'
            peak_val = high
            peak_idx = i

# Handle last ongoing if any
if state == 'TROUGH':
    drop_days = trough_idx - peak_idx
    drop_pct = (trough_val - peak_val) / peak_val
    t_start = max(0, trough_idx - 5)
    t_end = min(len(df)-1, trough_idx + 5)
    period_df = df.iloc[t_start:t_end+1]
    vix_max = period_df['VIX'].max()
    rsi60_min = period_df['RSI60'].min()
    rsi14_min = period_df['RSI14'].min()
    trough_row = df.iloc[trough_idx]
    isGrind = trough_row['Low'] <= trough_row['MA240'] * 1.05 and rsi60_min <= 46 and vix_max <= 28
    isStructural = vix_max > 28 and rsi60_min <= 42
    isFlash = vix_max > 35 and rsi14_min <= 30
    label = []
    if isStructural: label.append('🟡 長線恐慌底')
    if isFlash: label.append('🔵 閃崩底')
    if isGrind and not isStructural: label.append('🟢 陰跌支撐底')
    if not label: label = ['⚪ 無極端訊號']
    periods.append({
        'Period': f"{df['DateStr'].iloc[peak_idx]} ~ {df['DateStr'].iloc[trough_idx]} (持續中)",
        'Drop': f"{drop_pct*100:.2f}%",
        'Days': drop_days,
        'Speed': f"{(drop_pct/drop_days)*100 if drop_days > 0 else drop_pct*100:.2f}%/天",
        'VIX_Peak': f"{vix_max:.2f}",
        'RSI60_Low': f"{rsi60_min:.2f}",
        'Signal': ' + '.join(label)
    })

df_out = pd.DataFrame(periods)
with open('report.md', 'w') as f:
    f.write(df_out.to_markdown(index=False))
