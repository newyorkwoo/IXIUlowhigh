"""
MCI 2.0：自創的「複合投降分數 (Composite Capitulation Score, CCS)」

第一版嘗試(逐日、跟所有交易日比 VIX/RSI 絕對水位百分位)已經試過兩種寫法，
結果都無法把「真底部」和「普通回檔」分開——因為任何 >10% 回檔的谷底那天，
不管回檔本身有多淺，在「自己波段內」都會是相對極值，逐日算分沒辦法區分
「這次回檔本身有多嚴重」。改成波段(episode)等級的排名分數之後才真正可用，
細節記在下面。

舊版 MCI 的已知問題(這次分析實測驗證出來的):
  1. 用固定絕對門檻 (VIX>=33 / RSI60<=43.5)。在低波動年代(如 2004-2007)門檻
     太高，2007/10~2008/3 那種慢慢陰跌的空頭中繼，VIX 最高只衝到 32.24，
     永遠不會觸發兩道門檻。
  2. MCI 分數對「瞬間恐慌高度」極敏感 (VIX 用平方級的 fear_premium，RSI 用平方的
     oversold_exhaustion)，但完全沒有跟「歷史上其他波段」比較過，所以 2018/8、
     2021/11、2022/8 這三個事後證明是好買點的波段，MCI 只有 8~14 分，會被
     系統當成「醞釀中」只給 10-20% 小倉位。
  3. 只看恐慌的「瞬間最高值」，沒有把「跌了多久」變成分數的一部分，2002/12~2003/3
     這種 190 天的緩跌拿不到高分。

CCS 的設計：不逐日算分，而是在「波段(每次 >10% 回檔)結束時」用 5 個特徵在
「歷史上所有波段」裡的百分位排名，取平均當作這次波段的嚴重程度分數 (0~100)：
  P_vix      : 波段內 VIX 最高點，在歷史波段中的百分位
  P_rsi      : 波段內 RSI60 最低點(取反)，在歷史波段中的百分位
  P_damage   : price_damage = |跌幅%| / sqrt(下跌天數)，在歷史波段中的百分位
  P_duration : 下跌天數，在歷史波段中的百分位 (讓緩跌也能靠「跌很久」拿分)
  P_accel    : 末端 5 日 VIX 上升幅度，在歷史波段中的百分位 (末端是否還在加速恐慌)

用「歷史所有波段」當比較母體，而不是「所有交易日」，這樣才是同一個量級的
比較(波段跟波段比，而不是波段跟平靜盤整比)，分離度才會出來。

即時使用時(波段還沒走完)，用「目前波段的截至目前為止的統計值」去對「過去已經
走完的波段」做百分位排名 —— 不會用到未來資料，可以安全地用在即時儀表板上。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from analysis.classify_episodes import load_data, extract_periods, mci_score, classify


def _percentile_rank(value, population):
    population = sorted(population)
    if not population:
        return 0.5
    import bisect
    pos = bisect.bisect_left(population, value)
    return pos / len(population)


def compute_episode_ccs(periods, walk_forward=True):
    """幫每個歷史波段算 CCS。walk_forward=True 時，第 i 個波段只跟「在它之前
    已經結束」的波段比較，避免用到未來資訊(適合驗證這指標拿去做即時判斷時是否有效)。
    """
    vix_pool, rsi_pool, damage_pool, duration_pool, accel_pool = [], [], [], [], []

    for p in periods:
        price_damage = (abs(p['drop_pct']) * 100) / max(1, p['drop_days']) ** 0.5
        vix_accel = p['final_vix_5d_end'] - p['final_vix_5d_start']

        if walk_forward:
            p_vix = _percentile_rank(p['vix_max'], vix_pool)
            p_rsi = _percentile_rank(-p['rsi_min'], [-x for x in rsi_pool])
            p_damage = _percentile_rank(price_damage, damage_pool)
            p_duration = _percentile_rank(p['drop_days'], duration_pool)
            p_accel = _percentile_rank(vix_accel, accel_pool)
        else:
            p_vix = p_rsi = p_damage = p_duration = p_accel = None  # 填在下面第二輪

        p['price_damage'] = price_damage
        p['vix_accel'] = vix_accel
        p['ccs_walk_forward'] = 100 * (p_vix + p_rsi + p_damage + p_duration + p_accel) / 5 if walk_forward else None

        vix_pool.append(p['vix_max'])
        rsi_pool.append(p['rsi_min'])
        damage_pool.append(price_damage)
        duration_pool.append(p['drop_days'])
        accel_pool.append(vix_accel)

    # 全樣本版(用全部 40 個波段互相比較，僅供離線分析看分離度，不能拿去即時用)
    all_vix = [p['vix_max'] for p in periods]
    all_rsi = [-p['rsi_min'] for p in periods]
    all_damage = [p['price_damage'] for p in periods]
    all_duration = [p['drop_days'] for p in periods]
    all_accel = [p['vix_accel'] for p in periods]

    for p in periods:
        p_vix = _percentile_rank(p['vix_max'], all_vix)
        p_rsi = _percentile_rank(-p['rsi_min'], all_rsi)
        p_damage = _percentile_rank(p['price_damage'], all_damage)
        p_duration = _percentile_rank(p['drop_days'], all_duration)
        p_accel = _percentile_rank(p['vix_accel'], all_accel)
        p['ccs_full_sample'] = 100 * (p_vix + p_rsi + p_damage + p_duration + p_accel) / 5

    return periods


if __name__ == '__main__':
    df = load_data()
    periods = extract_periods(df)
    for p in periods:
        p['mci'] = mci_score(p)
        p['label'] = classify(p)

    periods = compute_episode_ccs(periods)

    print("=" * 108)
    print(f"{'波段':<24}{'標籤':<10}{'舊MCI':>8}{'CCS(全樣本)':>14}{'CCS(walk-forward)':>20}")
    print("=" * 108)
    for p in periods:
        wf = p['ccs_walk_forward']
        wf_str = f"{wf:.1f}" if wf is not None else "n/a(樣本太少)"
        print(f"{p['start_date']}~{p['end_date']:<10}  {p['label']:<8}{p['mci']:>8.1f}{p['ccs_full_sample']:>14.1f}{wf_str:>20}")

    bottoms = [p for p in periods if p['label'] in ('波段底部', '史詩大底')]
    non_bottoms = [p for p in periods if p['label'] not in ('波段底部', '史詩大底')]

    print("\n--- 分類分離度比較 (全樣本 CCS) ---")
    print(f"真底部 (n={len(bottoms)})  CCS 範圍: {min(p['ccs_full_sample'] for p in bottoms):.1f} ~ {max(p['ccs_full_sample'] for p in bottoms):.1f}")
    print(f"非底部 (n={len(non_bottoms)})  CCS 範圍: {min(p['ccs_full_sample'] for p in non_bottoms):.1f} ~ {max(p['ccs_full_sample'] for p in non_bottoms):.1f}")
    print(f"舊 MCI －－ 真底部範圍: {min(p['mci'] for p in bottoms):.1f} ~ {max(p['mci'] for p in bottoms):.1f}   非底部範圍: {min(p['mci'] for p in non_bottoms):.1f} ~ {max(p['mci'] for p in non_bottoms):.1f}")

    print("\n--- 舊系統誤判過的 3 個波段 (MCI<15 但事後是好買點)，新 CCS 排名如何？ ---")
    for key in ['2018-08-30~2018-12-24', '2021-11-22~2022-06-16', '2022-08-16~2022-10-13']:
        for p in periods:
            if f"{p['start_date']}~{p['end_date']}" == key:
                print(f"  {key}: 舊MCI={p['mci']:.1f}  CCS(全樣本)={p['ccs_full_sample']:.1f}  CCS(walk-forward)={p['ccs_walk_forward']:.1f}")

    print("\n--- 舊系統完全漏接的陰跌型盲點，新 CCS 排名如何？ ---")
    for key in ['2002-12-02~2003-03-12', '2007-10-31~2008-03-17']:
        for p in periods:
            if f"{p['start_date']}~{p['end_date']}" == key:
                print(f"  {key}: 舊MCI={p['mci']:.1f}(標籤={p['label']})  CCS(全樣本)={p['ccs_full_sample']:.1f}  CCS(walk-forward)={p['ccs_walk_forward']:.1f}")

    out_rows = [{
        'period': f"{p['start_date']}~{p['end_date']}", 'label': p['label'], 'mci': p['mci'],
        'ccs_full_sample': p['ccs_full_sample'], 'ccs_walk_forward': p['ccs_walk_forward'],
    } for p in periods]
    pd.DataFrame(out_rows).to_csv('analysis/mci2_comparison.csv', index=False)
    print("\n已輸出 analysis/mci2_comparison.csv")
