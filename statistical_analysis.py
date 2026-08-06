import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 1. Fetch and Prep Data
df = yf.Ticker('^IXIC').history(start='2000-01-01')
vix_df = yf.Ticker('^VIX').history(start='2000-01-01')

if hasattr(vix_df.index, 'tz') and vix_df.index.tz is not None:
    vix_df.index = vix_df.index.tz_localize(None)
if hasattr(df.index, 'tz') and df.index.tz is not None:
    df.index = df.index.tz_localize(None)

df = df.join(vix_df[['Close']], rsuffix='_VIX')
df.rename(columns={'Close_VIX': 'VIX'}, inplace=True)
df['VIX'] = df['VIX'].ffill().fillna(0)

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain60 = gain.ewm(alpha=1/60, min_periods=60, adjust=False).mean()
avg_loss60 = loss.ewm(alpha=1/60, min_periods=60, adjust=False).mean()
df['RSI60'] = 100 - (100 / (1 + avg_gain60 / avg_loss60))
df['RSI60'] = df['RSI60'].fillna(50)

df['ROC5'] = df['Close'].pct_change(periods=5) * 100
df['ROC5'] = df['ROC5'].fillna(0)
df.reset_index(inplace=True)

# 2. Find Drawdowns (>10%)
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
            drop_pct = (trough_val - peak_val) / peak_val * 100
            
            period_df = df.iloc[peak_idx:trough_idx+1]
            vix_max = period_df['VIX'].max()
            rsi_min = period_df['RSI60'].min()
            final_roc5 = df['ROC5'].iloc[trough_idx]
            
            periods.append({
                'Drop_Pct': drop_pct,
                'Drop_Days': drop_days,
                'Avg_Speed': drop_pct / drop_days if drop_days > 0 else drop_pct,
                'Final_ROC5': final_roc5,
                'VIX_Max': vix_max,
                'RSI60_Min': rsi_min
            })
            state = 'PEAK'
            peak_val = high
            peak_idx = i

p_df = pd.DataFrame(periods).dropna()

print("==== 1. 相關性矩陣 (Correlation Matrix) ====")
corr = p_df.corr().round(2)
print(corr)

print("\n==== 2. 極端崩盤 vs 一般回檔 的數據對比 ====")
# 極端崩盤: 跌幅大於 20%
extreme = p_df[p_df['Drop_Pct'] <= -20]
mild = p_df[p_df['Drop_Pct'] > -20]
print(f"[超級大熊市 (波段跌幅 > 20%), 發生 {len(extreme)} 次] 平均特徵:")
print(extreme[['Drop_Days', 'Final_ROC5', 'VIX_Max', 'RSI60_Min']].mean().round(2))
print(f"\n[一般波段回檔 (10% ~ 20%), 發生 {len(mild)} 次] 平均特徵:")
print(mild[['Drop_Days', 'Final_ROC5', 'VIX_Max', 'RSI60_Min']].mean().round(2))

print("\n==== 3. '最後5日趕底' 與 'VIX' 的交叉分析 ====")
panic_drops = p_df[p_df['Final_ROC5'] <= -8.0]
print(f"當最後5日跌幅超過 -8% 時 (發生 {len(panic_drops)} 次)，VIX 平均會噴到: {panic_drops['VIX_Max'].mean():.1f}")
print(f"此時 RSI60 平均會殺到: {panic_drops['RSI60_Min'].mean():.1f}")

