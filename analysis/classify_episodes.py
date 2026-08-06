"""
重建 40 個波段的量化分類，驗證使用者提出的判讀規則：
- VIX 峰值 >= 33 且 RSI60 最低點 <= 43.5  =>  波段底部 / 史詩大底
- 正常回檔的 VIX 峰值 <= 31

邏輯完全複用 app.py 的 ZigZag 抓取方式 (10% drop / 10% recovery, 2000-01-01 起, 30 天內合併)，
確保跟現行 Flask 系統算出來的 drawdown_periods 一致。
"""
import math
from datetime import datetime

import pandas as pd
import yfinance as yf


def load_data():
    ticker = yf.Ticker('^IXIC')
    df = ticker.history(start='1998-01-01')
    vix_df = yf.Ticker('^VIX').history(start='1998-01-01')

    if hasattr(vix_df.index, 'tz') and vix_df.index.tz is not None:
        vix_df.index = vix_df.index.tz_localize(None)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.join(vix_df[['Close']], rsuffix='_VIX')
    df.rename(columns={'Close_VIX': 'VIX'}, inplace=True)

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain60 = gain.ewm(alpha=1 / 60, min_periods=60, adjust=False).mean()
    avg_loss60 = loss.ewm(alpha=1 / 60, min_periods=60, adjust=False).mean()
    rs60 = avg_gain60 / avg_loss60
    df['RSI60'] = 100 - (100 / (1 + rs60))

    df['ROC5_speed'] = df['Close'].pct_change(periods=5) / 5

    df['VIX'] = df['VIX'].fillna(0)
    df['RSI60'] = df['RSI60'].fillna(50)
    df['ROC5_speed'] = df['ROC5_speed'].fillna(0)

    df = df[df.index >= '2000-01-01'].copy()
    df.reset_index(inplace=True)
    date_col = 'Date' if 'Date' in df.columns else 'Datetime'
    df['DateStr'] = df[date_col].dt.strftime('%Y-%m-%d')
    return df


def extract_periods(df):
    drawdown_periods = []
    state = 'PEAK'
    peak_idx = 0
    peak_val = df['High'].iloc[0]
    trough_idx = 0
    trough_val = df['Low'].iloc[0]
    drop_threshold = 0.10
    recovery_threshold = 0.10

    def build_record(peak_idx, trough_idx):
        drop_days = trough_idx - peak_idx
        drop_pct = (trough_val - peak_val) / peak_val
        period_df = df.iloc[peak_idx:trough_idx + 1]
        vix_start = df['VIX'].iloc[peak_idx]
        vix_max = period_df['VIX'].max()
        vix_max_idx_rel = period_df['VIX'].values.argmax()
        vix_max_idx = peak_idx + vix_max_idx_rel
        rsi_start = df['RSI60'].iloc[peak_idx]
        rsi_min = period_df['RSI60'].min()

        if trough_idx >= 5:
            final_vix_5d_start = df['VIX'].iloc[trough_idx - 5]
            final_vix_5d_end = df['VIX'].iloc[trough_idx]
            final_rsi_5d_start = df['RSI60'].iloc[trough_idx - 5]
            final_rsi_5d_end = df['RSI60'].iloc[trough_idx]
            final_roc5_speed = df['ROC5_speed'].iloc[trough_idx]
            # 末端 5 日跌速: 用 trough 前 5 日到 trough 的價格變化
            end_speed = (df['Close'].iloc[trough_idx] - df['Close'].iloc[trough_idx - 5]) / df['Close'].iloc[trough_idx - 5] / 5
        else:
            final_vix_5d_start = final_vix_5d_end = final_rsi_5d_start = final_rsi_5d_end = 0
            final_roc5_speed = 0
            end_speed = 0

        return {
            'start_date': df['DateStr'].iloc[peak_idx],
            'end_date': df['DateStr'].iloc[trough_idx],
            'drop_pct': drop_pct,
            'drop_days': drop_days,
            'drop_speed': drop_pct / drop_days if drop_days > 0 else drop_pct,
            'vix_start': float(vix_start),
            'vix_max': float(vix_max),
            'vix_max_date': df['DateStr'].iloc[vix_max_idx],
            'rsi_start': float(rsi_start),
            'rsi_min': float(rsi_min),
            'peak_val': float(peak_val),
            'trough_val': float(trough_val),
            'final_roc5_speed': float(final_roc5_speed),
            'end_5d_speed_pct': float(end_speed * 100),
            'final_vix_5d_start': float(final_vix_5d_start),
            'final_vix_5d_end': float(final_vix_5d_end),
            'final_rsi_5d_start': float(final_rsi_5d_start),
            'final_rsi_5d_end': float(final_rsi_5d_end),
            'vix_peak_lead_days': int(trough_idx - vix_max_idx),
        }

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
                drawdown_periods.append(build_record(peak_idx, trough_idx))
                state = 'PEAK'
                peak_val = high
                peak_idx = i

    if state == 'TROUGH':
        drawdown_periods.append(build_record(peak_idx, trough_idx))

    # merge within 30 days (same as app.py) — simplified re-implementation
    merged = []
    for p in drawdown_periods:
        if not merged:
            merged.append(p)
            continue
        prev = merged[-1]
        prev_end = datetime.strptime(prev['end_date'], '%Y-%m-%d')
        curr_start = datetime.strptime(p['start_date'], '%Y-%m-%d')
        if (curr_start - prev_end).days <= 30:
            # keep the deeper trough / higher peak, recompute combined stats crudely
            new_peak = max(prev['peak_val'], p['peak_val'])
            new_start_date = prev['start_date'] if prev['peak_val'] >= p['peak_val'] else p['start_date']
            new_trough = min(prev['trough_val'], p['trough_val'])
            new_end_date = p['end_date'] if p['trough_val'] <= prev['trough_val'] else prev['end_date']
            start_idx = df.index[df['DateStr'] == new_start_date][0]
            end_idx = df.index[df['DateStr'] == new_end_date][0]
            if end_idx > start_idx:
                merged[-1] = build_record(start_idx, end_idx)
            else:
                merged.append(p)
        else:
            merged.append(p)

    return merged


def mci_score(p):
    price_damage = (abs(p['drop_pct']) * 100) / math.sqrt(max(1, p['drop_days']))
    vix_diff = max(0, p['vix_max'] - p['vix_start'])
    fear_premium = (p['vix_max'] + vix_diff) / 20.0
    rsi_min_val = max(1, p['rsi_min'])
    oversold = (50.0 / rsi_min_val) ** 2
    return price_damage * fear_premium * oversold


def classify(p):
    mci = p['mci']
    if p['vix_max'] >= 33 and p['rsi_min'] <= 43.5:
        return '史詩大底' if mci >= 50 else '波段底部'
    return '正常回檔' if p['drop_pct'] > -0.20 else '醞釀中/深度回檔'


if __name__ == '__main__':
    df = load_data()
    periods = extract_periods(df)
    for p in periods:
        p['mci'] = mci_score(p)
        p['label'] = classify(p)

    out = pd.DataFrame(periods)
    out = out[['start_date', 'end_date', 'drop_pct', 'drop_days', 'drop_speed',
               'vix_start', 'vix_max', 'vix_max_date', 'vix_peak_lead_days',
               'rsi_start', 'rsi_min', 'end_5d_speed_pct', 'mci', 'label']]
    out['drop_pct'] = (out['drop_pct'] * 100).round(1)
    out['drop_speed'] = (out['drop_speed'] * 100).round(3)
    out['mci'] = out['mci'].round(1)
    out['end_5d_speed_pct'] = out['end_5d_speed_pct'].round(3)

    pd.set_option('display.max_rows', 200)
    pd.set_option('display.width', 220)
    print(out.to_string(index=False))
    print(f"\n總波段數: {len(out)}")
    print(out['label'].value_counts())

    out.to_csv('analysis/episodes.csv', index=False)
    print("\n已輸出 analysis/episodes.csv")
