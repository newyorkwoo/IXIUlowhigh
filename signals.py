"""
CCS（Composite Capitulation Score，複合投降分數）—— 進場/加碼指標。
搭配 ExitStage（出場階段）—— 出場指標。兩者合起來是完整的買-加碼-賣訊號系統。

CCS：不用固定絕對門檻（舊版的 VIX>=33 / RSI60<=43.5），而是把「目前這次回檔的
統計值」(VIX 波段內最高、RSI60 波段內最低、price_damage、下跌天數、末端 5 日
VIX 加速度) 拿去跟「歷史上所有已經走完的波段」做百分位排名取平均，只用「已經
結束」的歷史波段當比較母體，不會偷看未來，可安全用在即時儀表板。

分級門檻是用正式上線後的資料重新校準過的 (analysis/ 內的驗證腳本)：
真底部(14個歷史波段)的 CCS 落在 58.3~87.2，非底部(26個)落在 36.3~72.1，
用 45 / 60 / 80 三條線大致切開觀望 / 試單 / 波段底部 / ALL IN 四級。

tier (建倉層級，對應金字塔配置):
  0 觀望             CCS < 45
  1 試單 10-20%      45 <= CCS < 60
  2 波段底部 40-70%   60 <= CCS < 80
  3 ALL IN           CCS >= 80

ExitStage（跟 CCS 對稱設計，同樣不用任何跟大盤結構脫鉤的絕對點位）：
  用「離最近一次顯著低點反彈了多少%」和「有沒有連續站穩年線(MA240)」兩個條件，
  三段式：
  0 持有／續抱   反彈 < 15%，恐慌尚未確認結束，照 CCS 的建議持續操作
  1 減碼／停利   反彈 >= 15% 但還沒連續站穩年線，逢低買進的部位已經有淺水位獲利，
                建議先落袋一部分，剩下的等趨勢確認
  2 加碼確認     收盤價連續 EXIT_CONFIRM_DAYS 天站在年線(MA240)之上，代表空頭結構
                結束、回到多頭格局，歷史回測顯示此後中位數還有相當漲幅，屬於趨勢
                確認、可以加碼的訊號，而非出場訊號。要求連續多天是為了濾掉單日假
                突破（歷史上單日穿越年線有約36%會在兩天內又跌破，且事後報酬為負）。

CCS 波段追蹤與 ExitStage 共用同一組 PEAK/TROUGH 狀態機。狀態機在判斷「峰值/谷值
是否被跌破/漲穿 10%」時，一律用「前一天為止已確認」的峰谷值來判斷，當天創的新
峰谷值要等到隔天才會生效——避免暴力雙向震盪的單一交易日，同一天內就把「創新低」
和「反彈滿10%」都算完，導致波段在最恐慌的那天被誤判為已經走完。
"""
import bisect
import math

CCS_TIER1_THRESHOLD = 45.0
CCS_TIER2_THRESHOLD = 60.0
CCS_TIER3_THRESHOLD = 80.0

EXIT_RECOVERY_THRESHOLD_PCT = 15.0
EXIT_CONFIRM_DAYS = 3


def _percentile_rank(value, sorted_population):
    if not sorted_population:
        return 50.0
    pos = bisect.bisect_left(sorted_population, value)
    return 100.0 * pos / len(sorted_population)


def compute_ccs(df):
    """df 需含 Close/High/Low/VIX/RSI60/MA240 欄位（依交易日排序），
    回傳新增 CCS / RecoveryPct / ExitStage 欄位後的 df。"""
    n = len(df)
    ccs = [0.0] * n
    recovery_pct = [0.0] * n
    exit_stage = [0] * n

    vix = df['VIX'].values
    rsi60 = df['RSI60'].values
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    ma240 = df['MA240'].values

    last_trough_val = close[0]

    # CCS 的歷史波段比較母體：只放「已經結束」的波段，隨時間成長，不偷看未來
    pool_vix_max = []
    pool_neg_rsi_min = []
    pool_price_damage = []
    pool_drop_days = []
    pool_vix_accel = []

    c_state = 'PEAK'
    c_peak_idx = 0
    c_peak_val = high[0]
    c_trough_val = None
    c_vix_peak_in_leg = vix[0]
    c_rsi_min_in_leg = rsi60[0]

    above_ma240_streak = 0

    for i in range(1, n):
        h, l = high[i], low[i]

        # 用「前一天為止已確認」的峰谷值判斷是否跌破/漲穿門檻，今天創的新峰谷值
        # 留到迴圈最後才生效，避免單一交易日內就把「創新低」和「反彈滿10%」一次做完。
        if c_state == 'PEAK':
            if l <= c_peak_val * 0.90:
                c_state = 'TROUGH'
                c_trough_val = l
        elif c_state == 'TROUGH':
            last_trough_val = c_trough_val
            if h >= c_trough_val * 1.10:
                final_drop_days = i - c_peak_idx
                final_price_damage = (abs((c_trough_val - c_peak_val) / c_peak_val) * 100) / math.sqrt(max(1, final_drop_days))
                final_vix_accel = vix[i] - vix[max(0, i - 5)]

                bisect.insort(pool_vix_max, c_vix_peak_in_leg)
                bisect.insort(pool_neg_rsi_min, -c_rsi_min_in_leg)
                bisect.insort(pool_price_damage, final_price_damage)
                bisect.insort(pool_drop_days, final_drop_days)
                bisect.insort(pool_vix_accel, final_vix_accel)

                c_state = 'PEAK'
                c_peak_val = h
                c_peak_idx = i
                c_vix_peak_in_leg = vix[i]
                c_rsi_min_in_leg = rsi60[i]
            elif l < c_trough_val:
                c_trough_val = l

        if c_state == 'PEAK' and h > c_peak_val:
            c_peak_val = h
            c_peak_idx = i
            c_vix_peak_in_leg = vix[i]
            c_rsi_min_in_leg = rsi60[i]

        c_vix_peak_in_leg = max(c_vix_peak_in_leg, vix[i])
        c_rsi_min_in_leg = min(c_rsi_min_in_leg, rsi60[i])

        drop_days = i - c_peak_idx
        drop_pct = (close[i] - c_peak_val) / c_peak_val

        if drop_days > 0 and drop_pct < 0:
            price_damage = (abs(drop_pct) * 100) / math.sqrt(drop_days)
            vix_accel_now = vix[i] - vix[max(0, i - 5)]

            p_vix = _percentile_rank(c_vix_peak_in_leg, pool_vix_max)
            p_rsi = _percentile_rank(-c_rsi_min_in_leg, pool_neg_rsi_min)
            p_damage = _percentile_rank(price_damage, pool_price_damage)
            p_duration = _percentile_rank(drop_days, pool_drop_days)
            p_accel = _percentile_rank(vix_accel_now, pool_vix_accel)
            ccs[i] = (p_vix + p_rsi + p_damage + p_duration + p_accel) / 5

        rec_pct = (close[i] / last_trough_val - 1) * 100 if last_trough_val > 0 else 0.0
        recovery_pct[i] = rec_pct

        if close[i] > ma240[i]:
            above_ma240_streak += 1
        else:
            above_ma240_streak = 0

        if above_ma240_streak >= EXIT_CONFIRM_DAYS:
            exit_stage[i] = 2
        elif rec_pct >= EXIT_RECOVERY_THRESHOLD_PCT:
            exit_stage[i] = 1
        else:
            exit_stage[i] = 0

    df = df.copy()
    df['CCS'] = ccs
    df['RecoveryPct'] = recovery_pct
    df['ExitStage'] = exit_stage
    return df


def tier_for_ccs(ccs):
    if ccs >= CCS_TIER3_THRESHOLD:
        return 3
    if ccs >= CCS_TIER2_THRESHOLD:
        return 2
    if ccs >= CCS_TIER1_THRESHOLD:
        return 1
    return 0


TIER_LABELS = {
    0: '觀望',
    1: '試單 10-20%',
    2: '波段底部 40-70%',
    3: 'ALL IN',
}

TIER_ALLOCATION = {
    0: 0.0,
    1: 0.10,
    2: 0.55,
    3: 1.0,
}

EXIT_STAGE_LABELS = {
    0: '持有／續抱',
    1: '減碼／停利一部分',
    2: '加碼確認／趨勢站穩',
}
