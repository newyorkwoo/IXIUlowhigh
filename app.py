from flask import Flask, jsonify, render_template
import yfinance as yf
import pandas as pd

from signals import (
    compute_ccs,
    tier_for_ccs,
    TIER_LABELS,
    EXIT_STAGE_LABELS,
)

app = Flask(__name__)

_cached_response = None

def fetch_and_cache():
    global _cached_response
    print("正在更新指數資料...")
    _cached_response = build_response()
    print("資料更新完成。")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def data():
    if _cached_response is None:
        fetch_and_cache()
    return jsonify(_cached_response)

def build_response():
    # Fetch Nasdaq data from 1998 to allow 240-day MA and RSI60 to warm up
    ticker = yf.Ticker('^IXIC')
    df = ticker.history(start='1998-01-01')
    
    if df.empty:
        return {"kline": [], "drawdown_periods": []}
        
    # Download VIX and merge
    vix_df = yf.Ticker('^VIX').history(start='1998-01-01')
    
    # Try to align timezones or remove them before joining
    if hasattr(vix_df.index, 'tz') and vix_df.index.tz is not None:
        vix_df.index = vix_df.index.tz_localize(None)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    df = df.join(vix_df[['Close']], rsuffix='_VIX')
    df.rename(columns={'Close_VIX': 'VIX'}, inplace=True)
    
    df['MA60'] = df['Close'].rolling(window=60).mean().bfill()
    df['MA240'] = df['Close'].rolling(window=240).mean().bfill()
    
    # Calculate RSI60 using Wilder's Smoothing
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain60 = gain.ewm(alpha=1/60, min_periods=60, adjust=False).mean()
    avg_loss60 = loss.ewm(alpha=1/60, min_periods=60, adjust=False).mean()
    rs60 = avg_gain60 / avg_loss60
    df['RSI60'] = 100 - (100 / (1 + rs60))
    
    # Calculate RSI14
    avg_gain14 = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss14 = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs14 = avg_gain14 / avg_loss14
    df['RSI14'] = 100 - (100 / (1 + rs14))

    
    # Calculate 5-day and 7-day Rate of Change (Drop Speed)
    df['ROC5'] = df['Close'].pct_change(periods=5)
    df['ROC5_speed'] = df['ROC5'] / 5
    df['ROC7'] = df['Close'].pct_change(periods=7) * 100
    
    # Fill NAs
    df['VIX'] = df['VIX'].fillna(0)
    df['RSI60'] = df['RSI60'].fillna(50)
    df['RSI14'] = df['RSI14'].fillna(50)

    df['ROC5_speed'] = df['ROC5_speed'].fillna(0)
    df['ROC7'] = df['ROC7'].fillna(0)
    
    # Truncate data to start exactly from 2000-01-01 for rendering and logic
    df = df[df.index >= '2000-01-01'].copy()
    df.reset_index(inplace=True)
    
    # Ensure Date is string
    date_col = 'Date' if 'Date' in df.columns else 'Datetime'
    
    df['DateStr'] = df[date_col].dt.strftime('%Y-%m-%d')
    
    # Calculate ZigZag drawdown periods
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
                vix_start = df['VIX'].iloc[peak_idx]
                vix_max = period_df['VIX'].max()
                rsi_start = df['RSI60'].iloc[peak_idx]
                rsi_min = period_df['RSI60'].min()
                
                final_roc5_speed = df['ROC5_speed'].iloc[trough_idx]
                if trough_idx >= 5:
                    final_vix_5d_start = df['VIX'].iloc[trough_idx - 5]
                    final_vix_5d_end = df['VIX'].iloc[trough_idx]
                    final_rsi_5d_start = df['RSI60'].iloc[trough_idx - 5]
                    final_rsi_5d_end = df['RSI60'].iloc[trough_idx]
                else:
                    final_vix_5d_start = 0
                    final_vix_5d_end = 0
                    final_rsi_5d_start = 0
                    final_rsi_5d_end = 0
                
                drawdown_periods.append({
                    'start_date': df['DateStr'].iloc[peak_idx],
                    'end_date': df['DateStr'].iloc[trough_idx],
                    'drop_pct': drop_pct,
                    'drop_days': drop_days,
                    'drop_speed': drop_pct / drop_days if drop_days > 0 else drop_pct,
                    'vix_start': vix_start,
                    'vix_max': vix_max,
                    'rsi_start': rsi_start,
                    'rsi_min': rsi_min,
                    'peak_val': float(peak_val),
                    'trough_val': float(trough_val),
                    'final_roc5_speed': float(final_roc5_speed),
                    'final_vix_5d_start': float(final_vix_5d_start),
                    'final_vix_5d_end': float(final_vix_5d_end),
                    'final_rsi_5d_start': float(final_rsi_5d_start),
                    'final_rsi_5d_end': float(final_rsi_5d_end)
                })
                state = 'PEAK'
                peak_val = high
                peak_idx = i

    # Handle ongoing drawdown
    if state == 'TROUGH':
        drop_days = trough_idx - peak_idx
        drop_pct = (trough_val - peak_val) / peak_val
        
        period_df = df.iloc[peak_idx:trough_idx+1]
        vix_start = df['VIX'].iloc[peak_idx]
        vix_max = period_df['VIX'].max()
        rsi_start = df['RSI60'].iloc[peak_idx]
        rsi_min = period_df['RSI60'].min()
        
        final_roc5_speed = df['ROC5_speed'].iloc[trough_idx]
        if trough_idx >= 5:
            final_vix_5d_start = df['VIX'].iloc[trough_idx - 5]
            final_vix_5d_end = df['VIX'].iloc[trough_idx]
            final_rsi_5d_start = df['RSI60'].iloc[trough_idx - 5]
            final_rsi_5d_end = df['RSI60'].iloc[trough_idx]
        else:
            final_vix_5d_start = 0
            final_vix_5d_end = 0
            final_rsi_5d_start = 0
            final_rsi_5d_end = 0
            
        drawdown_periods.append({
            'start_date': df['DateStr'].iloc[peak_idx],
            'end_date': df['DateStr'].iloc[trough_idx],
            'drop_pct': drop_pct,
            'drop_days': drop_days,
            'drop_speed': drop_pct / drop_days if drop_days > 0 else drop_pct,
            'vix_start': vix_start,
            'vix_max': vix_max,
            'rsi_start': rsi_start,
            'rsi_min': rsi_min,
            'peak_val': float(peak_val),
            'trough_val': float(trough_val),
            'final_roc5_speed': float(final_roc5_speed),
            'final_vix_5d_start': float(final_vix_5d_start),
            'final_vix_5d_end': float(final_vix_5d_end),
            'final_rsi_5d_start': float(final_rsi_5d_start),
            'final_rsi_5d_end': float(final_rsi_5d_end)
        })
    
    # --- Post-processing: Merge dates that are too close (within 30 days) ---
    from datetime import datetime
    merged_periods = []
    for p in drawdown_periods:
        if not merged_periods:
            merged_periods.append(p)
        else:
            prev = merged_periods[-1]
            prev_end = datetime.strptime(prev['end_date'], '%Y-%m-%d')
            curr_start = datetime.strptime(p['start_date'], '%Y-%m-%d')
            
            # If they are within 30 days of each other, merge them
            if (curr_start - prev_end).days <= 30:
                if p['peak_val'] > prev['peak_val']:
                    new_peak = p['peak_val']
                    new_start_date = p['start_date']
                    new_vix_start = p['vix_start']
                    new_rsi_start = p['rsi_start']
                else:
                    new_peak = prev['peak_val']
                    new_start_date = prev['start_date']
                    new_vix_start = prev['vix_start']
                    new_rsi_start = prev['rsi_start']
                    
                if p['trough_val'] < prev['trough_val']:
                    new_trough = p['trough_val']
                    new_end_date = p['end_date']
                    new_final_roc5_speed = p['final_roc5_speed']
                    new_final_vix_5d_start = p['final_vix_5d_start']
                    new_final_vix_5d_end = p['final_vix_5d_end']
                    new_final_rsi_5d_start = p['final_rsi_5d_start']
                    new_final_rsi_5d_end = p['final_rsi_5d_end']
                else:
                    new_trough = prev['trough_val']
                    new_end_date = prev['end_date']
                    new_final_roc5_speed = prev['final_roc5_speed']
                    new_final_vix_5d_start = prev['final_vix_5d_start']
                    new_final_vix_5d_end = prev['final_vix_5d_end']
                    new_final_rsi_5d_start = prev['final_rsi_5d_start']
                    new_final_rsi_5d_end = prev['final_rsi_5d_end']
                
                start_dt = datetime.strptime(new_start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(new_end_date, '%Y-%m-%d')
                
                if end_dt > start_dt:
                    new_drop_pct = (new_trough - new_peak) / new_peak
                    
                    start_idx_series = df.index[df['DateStr'] == new_start_date]
                    end_idx_series = df.index[df['DateStr'] == new_end_date]
                    if not start_idx_series.empty and not end_idx_series.empty:
                        new_days = max(1, end_idx_series[0] - start_idx_series[0])
                    else:
                        new_days = max(1, (end_dt - start_dt).days)
                        
                    prev['start_date'] = new_start_date
                    prev['end_date'] = new_end_date
                    prev['peak_val'] = float(new_peak)
                    prev['trough_val'] = float(new_trough)
                    prev['drop_pct'] = float(new_drop_pct)
                    prev['drop_days'] = int(new_days)
                    prev['drop_speed'] = float(new_drop_pct / new_days)
                    prev['vix_start'] = float(new_vix_start)
                    prev['rsi_start'] = float(new_rsi_start)
                    prev['vix_max'] = float(max(prev['vix_max'], p['vix_max']))
                    prev['rsi_min'] = float(min(prev['rsi_min'], p['rsi_min']))
                    prev['final_roc5_speed'] = float(new_final_roc5_speed)
                    prev['final_vix_5d_start'] = float(new_final_vix_5d_start)
                    prev['final_vix_5d_end'] = float(new_final_vix_5d_end)
                    prev['final_rsi_5d_start'] = float(new_final_rsi_5d_start)
                    prev['final_rsi_5d_end'] = float(new_final_rsi_5d_end)
                else:
                    merged_periods.append(p)
            else:
                merged_periods.append(p)
                
    drawdown_periods = merged_periods
    # -------------------------------------------------------------------------
    
    # --- 計算 CCS (複合投降分數，唯一判讀指標) ---
    df = compute_ccs(df)
    df['Tier'] = [tier_for_ccs(ccs) for ccs in df['CCS']]

    ccs_by_date = dict(zip(df['DateStr'], df['CCS']))
    for p in drawdown_periods:
        p['ccs'] = float(ccs_by_date.get(p['end_date'], 0.0))

    kline_data = []
    for _, row in df.iterrows():
        kline_data.append({
            'date': row['DateStr'],
            'open': row['Open'],
            'close': row['Close'],
            'low': row['Low'],
            'high': row['High'],
            'vix': row['VIX'],
            'rsi60': row['RSI60'],
            'rsi14': row['RSI14'],
            'ma60': row['MA60'],
            'ma240': row['MA240'],
            'roc7': row['ROC7'],
            'ccs': row['CCS'],
            'tier': int(row['Tier']),
            'tier_label': TIER_LABELS[int(row['Tier'])],
            'recovery_pct': row['RecoveryPct'],
            'exit_stage': int(row['ExitStage']),
            'exit_stage_label': EXIT_STAGE_LABELS[int(row['ExitStage'])],
        })

    return {
        "kline": kline_data,
        "drawdown_periods": drawdown_periods
    }

if __name__ == '__main__':
    fetch_and_cache()
    app.run(debug=True, port=5000)
