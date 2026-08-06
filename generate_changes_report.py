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
            drop_days = trough_idx - peak_idx
            drop_pct = (trough_val - peak_val) / peak_val
            
            period_df = df.iloc[peak_idx:trough_idx+1]
            
            vix_start = df['VIX'].iloc[peak_idx]
            vix_max = period_df['VIX'].max()
            rsi_start = df['RSI60'].iloc[peak_idx]
            rsi_min = period_df['RSI60'].min()
            
            # Re-evaluate signals at trough + 5 buffer for minimums
            t_start = max(0, trough_idx - 5)
            t_end = min(len(df)-1, trough_idx + 5)
            buf_df = df.iloc[t_start:t_end+1]
            buf_vix_max = buf_df['VIX'].max()
            buf_rsi60_min = buf_df['RSI60'].min()
            buf_rsi14_min = buf_df['RSI14'].min()
            buf_roc7_min = buf_df['ROC7'].min()
            
            isStructural = buf_vix_max > 28 and buf_rsi60_min <= 42
            isFlash = buf_vix_max > 35 and buf_rsi14_min <= 30
            isExtreme = buf_roc7_min <= -8.0
            
            labels = []
            if isStructural: labels.append('🟡 恐慌')
            if isFlash: labels.append('🔵 閃崩')
            if isExtreme: labels.append('🟣 動能')
            if not labels: labels.append('⚪ 無')
            
            periods.append({
                '波段區間': f"{df['DateStr'].iloc[peak_idx]} ~ {df['DateStr'].iloc[trough_idx]}",
                '波段跌幅': f"{drop_pct*100:.1f}%",
                '下跌天數': drop_days,
                'VIX 變化 (起跌->極值)': f"{vix_start:.1f} 飆至 {vix_max:.1f}",
                'RSI60 變化 (起跌->極值)': f"{rsi_start:.1f} 殺至 {rsi_min:.1f}",
                '觸發神針': ' + '.join(labels)
            })
            state = 'PEAK'
            peak_val = high
            peak_idx = i

# ongoing
if state == 'TROUGH':
    drop_days = trough_idx - peak_idx
    drop_pct = (trough_val - peak_val) / peak_val
    period_df = df.iloc[peak_idx:trough_idx+1]
    vix_start = df['VIX'].iloc[peak_idx]
    vix_max = period_df['VIX'].max()
    rsi_start = df['RSI60'].iloc[peak_idx]
    rsi_min = period_df['RSI60'].min()
    
    t_start = max(0, trough_idx - 5)
    t_end = min(len(df)-1, trough_idx + 5)
    buf_df = df.iloc[t_start:t_end+1]
    buf_vix_max = buf_df['VIX'].max()
    buf_rsi60_min = buf_df['RSI60'].min()
    buf_rsi14_min = buf_df['RSI14'].min()
    buf_roc7_min = buf_df['ROC7'].min()
    
    isStructural = buf_vix_max > 28 and buf_rsi60_min <= 42
    isFlash = buf_vix_max > 35 and buf_rsi14_min <= 30
    isExtreme = buf_roc7_min <= -8.0
    labels = []
    if isStructural: labels.append('🟡 恐慌')
    if isFlash: labels.append('🔵 閃崩')
    if isExtreme: labels.append('🟣 動能')
    if not labels: labels.append('⚪ 無')
    
    periods.append({
        '波段區間': f"{df['DateStr'].iloc[peak_idx]} ~ {df['DateStr'].iloc[trough_idx]}",
        '波段跌幅': f"{drop_pct*100:.1f}%",
        '下跌天數': drop_days,
        'VIX 變化 (起跌->極值)': f"{vix_start:.1f} 飆至 {vix_max:.1f}",
        'RSI60 變化 (起跌->極值)': f"{rsi_start:.1f} 殺至 {rsi_min:.1f}",
        '觸發神針': ' + '.join(labels)
    })

df_out = pd.DataFrame(periods)
with open('drawdown_changes.md', 'w', encoding='utf-8') as f:
    f.write("# 那斯達克歷史回撤 ( >10% ) VIX 與 RSI 變化全紀錄\n\n")
    f.write(df_out.to_markdown(index=False))
