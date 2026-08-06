import yfinance as yf
import pandas as pd
import math
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# ================= 配置區 =================
# 建議您使用 Google 的「應用程式密碼 (App Password)」來發信，不要使用真實的登入密碼
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "your_email@gmail.com")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "your_app_password_here")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL", "your_email@gmail.com")

# 設定觸發通知的最低分數 (大於 5 才會收到 Email)
MCI_ALERT_THRESHOLD = 5.0 
# ==========================================

def calculate_realtime_mci():
    # 抓取最近幾年的數據來找尋目前波段的最高點
    ticker = yf.Ticker('^IXIC')
    df = ticker.history(start='2020-01-01')
    
    if df.empty:
        return None
        
    # 抓取 VIX
    vix_df = yf.Ticker('^VIX').history(start='2020-01-01')
    
    # 處理時區問題
    if hasattr(vix_df.index, 'tz') and vix_df.index.tz is not None:
        vix_df.index = vix_df.index.tz_localize(None)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    df = df.join(vix_df[['Close']], rsuffix='_VIX')
    df.rename(columns={'Close_VIX': 'VIX'}, inplace=True)
    df['VIX'] = df['VIX'].ffill().fillna(0)
    
    # 計算 RSI60 (Wilder's Smoothing)
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
    
    # 尋找目前的波段狀態
    state = 'PEAK' 
    peak_idx = 0
    peak_val = df['High'].iloc[0]
    vix_start = df['VIX'].iloc[0]
    
    drop_threshold = 0.10
    recovery_threshold = 0.10
    
    for i in range(1, len(df)):
        high = df['High'].iloc[i]
        low = df['Low'].iloc[i]
        
        if state == 'PEAK':
            if high > peak_val:
                peak_val = high
                peak_idx = i
                vix_start = df['VIX'].iloc[i]
            
            if low <= peak_val * (1 - drop_threshold):
                state = 'TROUGH'
        
        elif state == 'TROUGH':
            # 如果反彈超過最低點的 10%，視為一個新波段的開始
            trough_val = df['Low'].iloc[peak_idx:i+1].min()
            if high >= trough_val * (1 + recovery_threshold):
                state = 'PEAK'
                peak_val = high
                peak_idx = i
                vix_start = df['VIX'].iloc[i]

    # 計算「今天 (最新一個交易日)」的即時數據
    last_idx = len(df) - 1
    current_date = df['DateStr'].iloc[last_idx]
    current_close = df['Close'].iloc[last_idx]
    current_vix = df['VIX'].iloc[last_idx]
    current_rsi = df['RSI60'].iloc[last_idx]
    
    drop_pct = (current_close - peak_val) / peak_val
    drop_days = last_idx - peak_idx
    
    # 如果目前在創新高，MCI = 0
    if drop_days <= 0 or drop_pct >= 0:
        return {
            'date': current_date,
            'mci': 0.0,
            'drop_pct': drop_pct,
            'drop_days': drop_days,
            'current_vix': current_vix,
            'current_rsi': current_rsi,
            'peak_val': peak_val,
            'peak_date': df['DateStr'].iloc[peak_idx]
        }
        
    # 計算即時 MCI 分數
    price_damage = (abs(drop_pct) * 100) / math.sqrt(max(1, drop_days))
    
    vix_max = df['VIX'].iloc[peak_idx:last_idx+1].max()
    vix_diff = max(0, vix_max - vix_start)
    fear_premium = (vix_max + vix_diff) / 20.0
    
    rsi_min = df['RSI60'].iloc[peak_idx:last_idx+1].min()
    rsi_min_val = max(1, rsi_min)
    oversold_exhaustion = (50.0 / rsi_min_val) ** 2
    
    mci = float(price_damage * fear_premium * oversold_exhaustion)
    
    return {
        'date': current_date,
        'mci': mci,
        'drop_pct': drop_pct,
        'drop_days': drop_days,
        'current_vix': current_vix,
        'vix_max': vix_max,
        'current_rsi': current_rsi,
        'rsi_min': rsi_min,
        'peak_val': peak_val,
        'peak_date': df['DateStr'].iloc[peak_idx]
    }

def send_email_alert(mci_data):
    mci = mci_data['mci']
    
    if mci < MCI_ALERT_THRESHOLD:
        print(f"[{mci_data['date']}] 當前 MCI ({mci:.2f}) 低於設定的警報閥值 ({MCI_ALERT_THRESHOLD})。不發送 Email。")
        return
        
    badge = "⚠️ 醞釀中"
    if mci >= 50:
        badge = "🔥 史詩大底"
    elif mci >= 15:
        badge = "📉 波段底部"
        
    subject = f"🚨 NASDAQ MCI 投降指數警報: {mci:.1f} ({badge})"
        
    body = f"""
    <html>
    <head></head>
    <body style="font-family: Arial, sans-serif; color: #333;">
        <h2 style="color: #2c3e50;">納斯達克 MCI 投降指數即時警報</h2>
        <p>資料日期: <strong>{mci_data['date']}</strong></p>
        <h1 style="color: #e74c3c;">當前 MCI 分數: {mci:.2f} ({badge})</h1>
        <hr style="border: 1px solid #eee;">
        <h3>📊 即時市場狀態分析：</h3>
        <ul>
            <li><b>前次高點:</b> {mci_data['peak_val']:.2f} (於 {mci_data['peak_date']})</li>
            <li><b>當前波段跌幅:</b> <span style="color: #e74c3c; font-weight: bold;">{mci_data['drop_pct']*100:.2f}%</span></li>
            <li><b>已下跌交易日:</b> {mci_data['drop_days']} 天</li>
            <li><b>期間最高 VIX:</b> {mci_data['vix_max']:.2f} (最新收盤: {mci_data['current_vix']:.2f})</li>
            <li><b>期間最低 RSI60:</b> {mci_data['rsi_min']:.2f} (最新收盤: {mci_data['current_rsi']:.2f})</li>
        </ul>
        <br>
        <p style="font-size: 0.9em; color: #7f8c8d;">這是一封由您的量化交易系統自動發送的即時通知。請根據您的交易策略進行風險控管與資金配置評估。</p>
    </body>
    </html>
    """
    
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    
    msg.attach(MIMEText(body, 'html'))
    
    try:
        # 預設使用 Google Gmail 的 SMTP 伺服器
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        # 登入信箱
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        # 寄送 email
        server.send_message(msg)
        server.quit()
        print(f"[{mci_data['date']}] 成功發送 MCI 警報通知信至 {RECEIVER_EMAIL}!")
    except Exception as e:
        print(f"發送 Email 失敗，請檢查信箱帳號密碼或網路設定。錯誤訊息: {e}")

if __name__ == "__main__":
    print("正在計算即時 MCI 投降指數...")
    data = calculate_realtime_mci()
    if data:
        print(f"最新交易日: {data['date']}")
        print(f"當前 MCI: {data['mci']:.2f} (累積跌幅: {data['drop_pct']*100:.2f}%)")
        
        # 預防密碼未設定直接執行導致報錯
        if SENDER_PASSWORD == "your_app_password_here":
            print("\n[注意] 您尚未設定 Gmail 應用程式密碼。請先在程式碼或環境變數中設定 SENDER_EMAIL 與 SENDER_PASSWORD 以啟用發信功能。")
        else:
            send_email_alert(data)
    else:
        print("無法取得市場數據，請檢查網路連線。")
