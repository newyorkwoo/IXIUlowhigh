"""
回測：MCI 不同門檻進場，統計各時間段報酬率與勝率
"""
import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import datetime

print("下載資料中...")
df = yf.Ticker('^IXIC').history(start='1998-01-01')
vix_df = yf.Ticker('^VIX').history(start='1998-01-01')

if hasattr(df.index, 'tz') and df.index.tz is not None:
    df.index = df.index.tz_localize(None)
if hasattr(vix_df.index, 'tz') and vix_df.index.tz is not None:
    vix_df.index = vix_df.index.tz_localize(None)

df = df.join(vix_df[['Close']], rsuffix='_VIX')
df.rename(columns={'Close_VIX': 'VIX'}, inplace=True)
df['VIX'] = df['VIX'].fillna(0)

# RSI60
delta = df['Close'].diff()
gain  = delta.where(delta > 0, 0)
loss  = -delta.where(delta < 0, 0)
avg_gain60 = gain.ewm(alpha=1/60, min_periods=60, adjust=False).mean()
avg_loss60 = loss.ewm(alpha=1/60, min_periods=60, adjust=False).mean()
df['RSI60'] = (100 - (100 / (1 + avg_gain60 / avg_loss60))).fillna(50)

df = df[df.index >= '2000-01-01'].copy()
df.reset_index(inplace=True)
date_col = 'Date' if 'Date' in df.columns else 'Datetime'
df['DateStr'] = df[date_col].dt.strftime('%Y-%m-%d')

# ── 計算 Daily MCI ─────────────────────────────────────────
df['Daily_MCI'] = 0.0
c_state = 'PEAK'
c_peak_idx = 0
c_peak_val = df['High'].iloc[0]
c_vix_start = df['VIX'].iloc[0]

for i in range(1, len(df)):
    h, lo = df['High'].iloc[i], df['Low'].iloc[i]
    if c_state == 'PEAK':
        if h > c_peak_val:
            c_peak_val, c_peak_idx, c_vix_start = h, i, df['VIX'].iloc[i]
        if lo <= c_peak_val * 0.90:
            c_state = 'TROUGH'
    else:
        trough_val = df['Low'].iloc[c_peak_idx:i+1].min()
        if h >= trough_val * 1.10:
            c_state = 'PEAK'
            c_peak_val, c_peak_idx, c_vix_start = h, i, df['VIX'].iloc[i]

    drop_days = i - c_peak_idx
    drop_pct  = (df['Close'].iloc[i] - c_peak_val) / c_peak_val
    if drop_days > 0 and drop_pct < 0:
        price_damage = (abs(drop_pct) * 100) / math.sqrt(drop_days)
        vix_cur = df['VIX'].iloc[i]
        fear_premium = (vix_cur + max(0, vix_cur - c_vix_start)) / 20.0
        rsi_cur = df['RSI60'].iloc[i]
        oversold = (50.0 / max(1, rsi_cur)) ** 2
        df.at[df.index[i], 'Daily_MCI'] = min(100.0, price_damage * fear_premium * oversold)

# ── 回測函式 ───────────────────────────────────────────────
def backtest(threshold, cooldown=20):
    """
    每次 MCI 超過 threshold，進場（收盤價），
    cooldown 天內不重複進場，統計各時間段報酬。
    """
    signals = []
    last_signal = -cooldown - 1

    for i in range(len(df)):
        if df['Daily_MCI'].iloc[i] >= threshold and (i - last_signal) > cooldown:
            signals.append(i)
            last_signal = i

    horizons = [20, 40, 60, 120, 252]
    results  = {h: [] for h in horizons}

    for sig_i in signals:
        entry_price = df['Close'].iloc[sig_i]
        for h in horizons:
            future_i = sig_i + h
            if future_i < len(df):
                ret = (df['Close'].iloc[future_i] - entry_price) / entry_price * 100
                results[h].append(ret)

    stats = {'threshold': threshold, 'signals': len(signals)}
    for h in horizons:
        rets = results[h]
        if rets:
            stats[f'avg_{h}d']  = round(np.mean(rets), 1)
            stats[f'win_{h}d']  = round(np.mean([r > 0 for r in rets]) * 100, 0)
            stats[f'med_{h}d']  = round(np.median(rets), 1)
        else:
            stats[f'avg_{h}d'] = stats[f'win_{h}d'] = stats[f'med_{h}d'] = None
    return stats

# ── 測試各門檻 ─────────────────────────────────────────────
thresholds = [1, 3, 5, 8, 10, 15, 20, 30, 50]
all_stats = [backtest(t) for t in thresholds]

# 基準：隨機進場（MCI≥0 = 每20天進場）
baseline = backtest(0, cooldown=20)

print()
print("=" * 100)
print(f"{'MCI門檻':>6}  {'進場次':>5}  "
      f"{'20日均%':>7} {'20勝%':>6}  "
      f"{'60日均%':>7} {'60勝%':>6}  "
      f"{'120日均%':>8} {'120勝%':>7}  "
      f"{'252日均%':>8} {'252勝%':>7}")
print("-" * 100)

def fmt(v): return f"{v:>+6.1f}" if v is not None else "   N/A"
def fmtw(v): return f"{v:>5.0f}%" if v is not None else "  N/A"

for s in all_stats:
    t = s['threshold']
    tag = ""
    if t == 5:  tag = " ← 試單"
    if t == 15: tag = " ← 加碼"
    if t == 50: tag = " ← ALL IN"
    print(f"  MCI≥{t:<3} {s['signals']:>4}次  "
          f"{fmt(s['avg_20d'])} {fmtw(s['win_20d'])}  "
          f"{fmt(s['avg_60d'])} {fmtw(s['win_60d'])}  "
          f"{fmt(s['avg_120d'])} {fmtw(s['win_120d'])}  "
          f"{fmt(s['avg_252d'])} {fmtw(s['win_252d'])}{tag}")

print()
print(f"  基準(隨機) {baseline['signals']:>4}次  "
      f"{fmt(baseline['avg_20d'])} {fmtw(baseline['win_20d'])}  "
      f"{fmt(baseline['avg_60d'])} {fmtw(baseline['win_60d'])}  "
      f"{fmt(baseline['avg_120d'])} {fmtw(baseline['win_120d'])}  "
      f"{fmt(baseline['avg_252d'])} {fmtw(baseline['win_252d'])}")

# ── 分批建倉策略回測 ──────────────────────────────────────
print()
print("=" * 60)
print("分批建倉策略：MCI≥5(30%) + MCI≥15(40%) + MCI≥50(30%)")
print("統計：若完整分三批，持有 252 日後總報酬")
print("-" * 60)

batch_results = []
i = 0
while i < len(df):
    if df['Daily_MCI'].iloc[i] >= 5:
        entry1_i = i
        entry1_p = df['Close'].iloc[i]
        # 找第二批（MCI≥15，至少5天後）
        entry2_i = entry2_p = None
        for j in range(i+5, min(i+120, len(df))):
            if df['Daily_MCI'].iloc[j] >= 15:
                entry2_i, entry2_p = j, df['Close'].iloc[j]
                break
        # 找第三批（MCI≥50，在第二批之後）
        entry3_i = entry3_p = None
        start3 = entry2_i+1 if entry2_i else i+5
        for j in range(start3, min(i+180, len(df))):
            if df['Daily_MCI'].iloc[j] >= 50:
                entry3_i, entry3_p = j, df['Close'].iloc[j]
                break

        # 計算 252 日後報酬
        exit_i = i + 252
        if exit_i < len(df):
            exit_p = df['Close'].iloc[exit_i]
            ret1 = (exit_p - entry1_p) / entry1_p * 100 * 0.30
            ret2 = (exit_p - entry2_p) / entry2_p * 100 * 0.40 if entry2_p else 0
            ret3 = (exit_p - entry3_p) / entry3_p * 100 * 0.30 if entry3_p else 0
            weight = 0.30 + (0.40 if entry2_p else 0) + (0.30 if entry3_p else 0)
            total_ret = (ret1 + ret2 + ret3) / weight * 100 / 100
            batch_results.append({
                'date': df['DateStr'].iloc[i],
                'mci': df['Daily_MCI'].iloc[i],
                'batches': 1 + (1 if entry2_p else 0) + (1 if entry3_p else 0),
                'ret_1yr': round(total_ret, 1),
                'exit_date': df['DateStr'].iloc[exit_i]
            })
        i += 40  # 冷卻40天
    else:
        i += 1

if batch_results:
    rets = [r['ret_1yr'] for r in batch_results]
    wins = [r for r in rets if r > 0]
    print(f"共觸發 {len(batch_results)} 次分批建倉機會")
    print(f"平均加權報酬（1年後）：{np.mean(rets):+.1f}%")
    print(f"勝率：{len(wins)/len(batch_results)*100:.0f}%")
    print(f"中位報酬：{np.median(rets):+.1f}%")
    print()
    print(f"{'進場日':^12} {'MCI':>5} {'批數':>4} {'1年後報酬':>9} {'出場日':^12}")
    print("-" * 50)
    for r in batch_results:
        flag = "✓" if r['ret_1yr'] > 0 else "✗"
        print(f"  {r['date']}  {r['mci']:>5.1f}  {r['batches']:>3}批  "
              f"{r['ret_1yr']:>+8.1f}%  {r['exit_date']}  {flag}")
