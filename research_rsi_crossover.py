"""
研究：RSI60 與 RSI_N 黃金交叉 vs 波段低點的距離
找出哪個 N 的交叉點最貼近歷史抄底位置
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import math

# ── 1. 取資料 ───────────────────────────────────────────────
print("下載資料中...")
df_raw = yf.Ticker('^IXIC').history(start='1998-01-01')
if hasattr(df_raw.index, 'tz') and df_raw.index.tz is not None:
    df_raw.index = df_raw.index.tz_localize(None)

delta = df_raw['Close'].diff()
gain  = delta.where(delta > 0, 0)
loss  = -delta.where(delta < 0, 0)

# RSI60 Wilder's EWM
avg_gain60 = gain.ewm(alpha=1/60, min_periods=60, adjust=False).mean()
avg_loss60 = loss.ewm(alpha=1/60, min_periods=60, adjust=False).mean()
df_raw['RSI60'] = 100 - (100 / (1 + avg_gain60 / avg_loss60))
df_raw['RSI60'] = df_raw['RSI60'].fillna(50)

# 截至 2000 年後
df = df_raw[df_raw.index >= '2000-01-01'].copy()
df.reset_index(inplace=True)
date_col = 'Date' if 'Date' in df.columns else 'Datetime'
df['DateStr'] = df[date_col].dt.strftime('%Y-%m-%d')

# ── 2. 計算波段低點 (ZigZag 10%) ────────────────────────────
state = 'PEAK'
peak_idx, peak_val = 0, df['High'].iloc[0]
trough_idx, trough_val = 0, df['Low'].iloc[0]
raw_periods = []

for i in range(1, len(df)):
    h, lo = df['High'].iloc[i], df['Low'].iloc[i]
    if state == 'PEAK':
        if h > peak_val:
            peak_val, peak_idx = h, i
        if lo <= peak_val * 0.90:
            state, trough_val, trough_idx = 'TROUGH', lo, i
    else:
        if lo < trough_val:
            trough_val, trough_idx = lo, i
        if h >= trough_val * 1.10:
            raw_periods.append({'start_date': df['DateStr'].iloc[peak_idx],
                                 'end_date':   df['DateStr'].iloc[trough_idx],
                                 'drop_pct':   (trough_val - peak_val) / peak_val})
            state, peak_val, peak_idx = 'PEAK', h, i

if state == 'TROUGH':
    raw_periods.append({'start_date': df['DateStr'].iloc[peak_idx],
                         'end_date':   df['DateStr'].iloc[trough_idx],
                         'drop_pct':   (trough_val - peak_val) / peak_val})

# 合併 30 天內相鄰區間
merged = []
for p in raw_periods:
    if not merged:
        merged.append(p); continue
    prev = merged[-1]
    d0 = datetime.strptime(prev['end_date'], '%Y-%m-%d')
    d1 = datetime.strptime(p['start_date'], '%Y-%m-%d')
    if (d1 - d0).days <= 30:
        if p['drop_pct'] < prev['drop_pct']:
            prev['end_date']  = p['end_date']
            prev['drop_pct']  = p['drop_pct']
    else:
        merged.append(p)

drawdown_troughs = [datetime.strptime(p['end_date'], '%Y-%m-%d') for p in merged]
print(f"波段低點共 {len(drawdown_troughs)} 個")

# ── 3. 回測各 RSI 週期 ──────────────────────────────────────
def signed_days_to_nearest_trough(cross_dt, troughs):
    best = min(troughs, key=lambda t: abs((cross_dt - t).days))
    return (cross_dt - best).days   # 正 = 交叉在低點之後（滯後）；負 = 交叉在低點之前（領先）

test_periods = [80, 100, 120, 150, 180, 200, 240, 300, 360, 480, 600]

results = []
for N in test_periods:
    # Simple rolling RSI_N (計算方法與 app.py RSI240 相同)
    # 需從原始 df_raw gain/loss 計算，保留足夠 warm-up
    avg_gain_n = gain.rolling(window=N, min_periods=N).mean()
    avg_loss_n = loss.rolling(window=N, min_periods=N).mean()
    rs_n = avg_gain_n / avg_loss_n.replace(0, np.nan)
    rsi_n_full = 100 - (100 / (1 + rs_n))
    rsi_n_full = rsi_n_full.fillna(50)

    # 對齊到截斷後的 df
    rsi_n = rsi_n_full[df_raw.index >= '2000-01-01'].values
    rsi60  = df['RSI60'].values

    # 偵測黃金交叉
    golden_crosses = []
    for i in range(1, len(df)):
        prev_d = rsi60[i-1] - rsi_n[i-1]
        curr_d = rsi60[i]   - rsi_n[i]
        if prev_d < 0 and curr_d >= 0:
            cross_dt = datetime.strptime(df['DateStr'].iloc[i], '%Y-%m-%d')
            golden_crosses.append(cross_dt)

    if not golden_crosses or not drawdown_troughs:
        results.append({'N': N, 'crosses': 0})
        continue

    offsets = [signed_days_to_nearest_trough(c, drawdown_troughs) for c in golden_crosses]
    offsets = np.array(offsets)

    # 統計
    within_20  = np.sum(np.abs(offsets) <= 20)
    within_40  = np.sum(np.abs(offsets) <= 40)
    after_only = offsets[offsets >= 0]   # 滯後（低點之後才交叉 = 確認信號）
    early_only = offsets[offsets < 0]    # 領先（提前交叉）

    results.append({
        'N':          N,
        'crosses':    len(golden_crosses),
        'pct_20d':    within_20 / len(golden_crosses) * 100,
        'pct_40d':    within_40 / len(golden_crosses) * 100,
        'avg_offset': np.mean(offsets),
        'med_offset': np.median(offsets),
        'avg_lag':    np.mean(after_only) if len(after_only) > 0 else np.nan,
        'lag_count':  len(after_only),
        'early_count':len(early_only),
    })

# ── 4. 從「波段低點」角度：各底部有沒有被交叉信號捕到 ───────
print()
print("=== 從波段低點角度：各底部被捕到的比率 ===")
print(f"{'RSI_N':>6}  {'交叉次':>5}  {'底部捕±20日':>10}  {'底部捕±40日':>10}  {'底部捕±60日':>10}  {'最近交叉中位':>12}  {'平均滯後天數':>12}")
print("-" * 90)

for r in results:
    if r['crosses'] == 0:
        print(f"  RSI{r['N']:>3}   0次  (資料不足)")
        continue
    N = r['N']
    # 重新計算各底部的最近交叉
    avg_gain_n = gain.rolling(window=N, min_periods=N).mean()
    avg_loss_n = loss.rolling(window=N, min_periods=N).mean()
    rs_n = avg_gain_n / avg_loss_n.replace(0, np.nan)
    rsi_n_full = (100 - (100 / (1 + rs_n))).fillna(50)
    rsi_n = rsi_n_full[df_raw.index >= '2000-01-01'].values
    rsi60v = df['RSI60'].values

    golden_cross_dts = []
    for i in range(1, len(df)):
        if rsi60v[i-1] < rsi_n[i-1] and rsi60v[i] >= rsi_n[i]:
            golden_cross_dts.append(datetime.strptime(df['DateStr'].iloc[i], '%Y-%m-%d'))

    if not golden_cross_dts:
        continue

    # 對每個底部找最近的黃金交叉
    trough_offsets = []
    for t in drawdown_troughs:
        nearest_cross = min(golden_cross_dts, key=lambda c: abs((c - t).days))
        trough_offsets.append((nearest_cross - t).days)

    trough_offsets = np.array(trough_offsets)
    c20 = np.sum(np.abs(trough_offsets) <= 20)
    c40 = np.sum(np.abs(trough_offsets) <= 40)
    c60 = np.sum(np.abs(trough_offsets) <= 60)
    total = len(drawdown_troughs)
    lag_only = trough_offsets[trough_offsets > 0]
    avg_lag = np.mean(lag_only) if len(lag_only) else float('nan')

    r['trough_c20'] = c20 / total * 100
    r['trough_c40'] = c40 / total * 100
    r['trough_c60'] = c60 / total * 100
    r['trough_med'] = np.median(trough_offsets)
    r['trough_avg_lag'] = avg_lag

    print(f"  RSI{N:>3}  {r['crosses']:>4}次  "
          f"{c20:>3}/{total}={c20/total*100:>4.0f}%      "
          f"{c40:>3}/{total}={c40/total*100:>4.0f}%      "
          f"{c60:>3}/{total}={c60/total*100:>4.0f}%      "
          f"{np.median(trough_offsets):>+10.0f}日      "
          f"{avg_lag:>+10.1f}日")

# 找最佳
valid = [r for r in results if 'trough_c40' in r]
best = max(valid, key=lambda r: (r['trough_c40'], -abs(r['trough_med'])))
print()
print(f"★ 最佳週期：RSI{best['N']}")
print(f"  ±20日內捕到 {best['trough_c20']:.0f}% 的波段底部")
print(f"  ±40日內捕到 {best['trough_c40']:.0f}% 的波段底部")
print(f"  交叉中位在底部後 {best['trough_med']:+.0f} 天（正=滯後=等底部確立才進場）")
print(f"  平均滯後 {best['trough_avg_lag']:.0f} 天")

# 印出最佳週期每個底部的命中細節
print()
print(f"=== RSI{best['N']} 各波段底部命中明細 ===")
N = best['N']
avg_gain_n = gain.rolling(window=N, min_periods=N).mean()
avg_loss_n = loss.rolling(window=N, min_periods=N).mean()
rsi_n_full = (100 - (100 / (1 + avg_gain_n / avg_loss_n.replace(0, np.nan)))).fillna(50)
rsi_n = rsi_n_full[df_raw.index >= '2000-01-01'].values
rsi60v = df['RSI60'].values
golden_cross_dts = []
for i in range(1, len(df)):
    if rsi60v[i-1] < rsi_n[i-1] and rsi60v[i] >= rsi_n[i]:
        golden_cross_dts.append(datetime.strptime(df['DateStr'].iloc[i], '%Y-%m-%d'))

print(f"{'波段低點':^12}  {'最近交叉日':^12}  {'偏移天數':^8}  {'訊號':^10}  {'跌幅':>6}")
print("-" * 60)
for p, t in zip(merged, drawdown_troughs):
    nearest = min(golden_cross_dts, key=lambda c: abs((c - t).days))
    offset = (nearest - t).days
    flag = "✓ 滯後確認" if 0 <= offset <= 40 else ("△ 提前" if -20 <= offset < 0 else "✗ 偏遠")
    print(f"  {p['end_date']:^12}  {nearest.strftime('%Y-%m-%d'):^12}  {offset:>+6}天  {flag:^12}  {p['drop_pct']*100:.1f}%")
