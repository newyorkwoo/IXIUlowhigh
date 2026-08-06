"""
金字塔式判讀規則回測。

策略：每日把「NASDAQ 曝險比例」朝當天 tier 對應的目標配置 (TIER_ALLOCATION) 調整
(允許買進也允許減碼，無交易成本、無滑價，收盤價成交)。跟 100% Buy & Hold 比較。

tier 目標配置定義在 signals.TIER_ALLOCATION:
  0 觀望 -> 0%
  1 試單 10-20% -> 15%
  2 波段底部 40-70% -> 55%
  3 ALL IN -> 100%
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from app import build_response
from signals import TIER_ALLOCATION, TIER_LABELS


CORE_WEIGHT = 0.7    # 核心倉位：永遠持有，不因訊號進出
RESERVE_WEIGHT = 0.3  # 逢低加碼子彈：依 tier 分批部署，只加碼不減碼


def run_backtest(core_weight=CORE_WEIGHT, reserve_weight=RESERVE_WEIGHT):
    resp = build_response()
    kline = pd.DataFrame(resp['kline'])
    kline['date'] = pd.to_datetime(kline['date'])
    kline = kline.sort_values('date').reset_index(drop=True)

    kline['target_alloc'] = kline['tier'].map(TIER_ALLOCATION)  # 純戰術：訊號決定 0~100% 曝險
    kline['ret'] = kline['close'].pct_change().fillna(0)

    n = len(kline)
    close = kline['close'].values
    ret = kline['ret'].values
    target_alloc = kline['target_alloc'].values

    strat_value = np.zeros(n)   # 純戰術策略：無訊號時空手，訊號出現才進場
    bh_value = np.zeros(n)      # 100% Buy & Hold
    cs_value = np.zeros(n)      # 核心+衛星：70% 永遠持有 + 30% 子彈依訊號分批「只加碼不減碼」
    exposure = np.zeros(n)
    cs_exposure = np.zeros(n)

    strat_value[0] = 1.0
    bh_value[0] = 1.0
    cs_value[0] = 1.0
    exposure[0] = target_alloc[0]

    reserve_state = 0.0  # 0~1，子彈部署比例，只在創新高後歸零重新累積
    running_max_close = close[0]
    cs_exposure[0] = core_weight + reserve_weight * reserve_state

    for i in range(1, n):
        r = ret[i]
        alloc_prev = exposure[i - 1]
        strat_value[i] = strat_value[i - 1] * (1 + alloc_prev * r)
        bh_value[i] = bh_value[i - 1] * (1 + r)
        exposure[i] = target_alloc[i]

        cs_alloc_prev = cs_exposure[i - 1]
        cs_value[i] = cs_value[i - 1] * (1 + cs_alloc_prev * r)

        running_max_close = max(running_max_close, close[i])
        if close[i] >= running_max_close:
            reserve_state = 0.0  # 波段完全收復失土，子彈補滿，準備迎接下一次回檔
        else:
            reserve_state = max(reserve_state, target_alloc[i])  # 只加碼、不因訊號降級而減碼
        cs_exposure[i] = core_weight + reserve_weight * reserve_state

    kline['strat_value'] = strat_value
    kline['bh_value'] = bh_value
    kline['cs_value'] = cs_value
    kline['exposure'] = exposure
    kline['cs_exposure'] = cs_exposure

    return kline


def max_drawdown(values):
    peak = np.maximum.accumulate(values)
    dd = (values - peak) / peak
    return dd.min()


def cagr(values, dates):
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    return (values[-1] / values[0]) ** (1 / years) - 1


def sharpe(returns_series, freq=252):
    r = returns_series
    if r.std() == 0:
        return 0.0
    return (r.mean() / r.std()) * np.sqrt(freq)


def summarize(kline):
    dates = kline['date']

    series = {
        '純戰術(0%起跳)': ('strat_value', 'exposure'),
        '核心70%+子彈30%': ('cs_value', 'cs_exposure'),
        'Buy & Hold': ('bh_value', None),
    }

    print("=" * 72)
    print(f"回測期間: {dates.iloc[0].date()} ~ {dates.iloc[-1].date()}  ({len(kline)} 個交易日)")
    print("=" * 72)

    rets = {}
    header = f"{'指標':<18}" + "".join(f"{name:>18}" for name in series)
    print(f"\n{header}")
    rows = {'總報酬': [], '年化報酬 CAGR': [], '最大回撤': [], '年化波動度': [], 'Sharpe(無息)': [], '平均曝險': []}
    for name, (val_col, exp_col) in series.items():
        v = kline[val_col].values
        r = kline[val_col].pct_change().fillna(0)
        rets[name] = r
        rows['總報酬'].append(f"{(v[-1]-1)*100:.1f}%")
        rows['年化報酬 CAGR'].append(f"{cagr(v, dates)*100:.2f}%")
        rows['最大回撤'].append(f"{max_drawdown(v)*100:.1f}%")
        rows['年化波動度'].append(f"{r.std()*np.sqrt(252)*100:.1f}%")
        rows['Sharpe(無息)'].append(f"{sharpe(r):.2f}")
        rows['平均曝險'].append(f"{kline[exp_col].mean()*100:.1f}%" if exp_col else "100.0%")
    for label, vals in rows.items():
        print(f"{label:<18}" + "".join(f"{v:>18}" for v in vals))

    print("\n各 tier 出現天數與占比 (純戰術/核心衛星共用同一組 tier 訊號):")
    tier_counts = kline['tier'].value_counts().sort_index()
    for t, c in tier_counts.items():
        print(f"  tier {t} ({TIER_LABELS[t]:<14}): {c:>5} 天  ({c/len(kline)*100:5.2f}%)")

    print("\n以年為單位分別看報酬:")
    kline['year'] = kline['date'].dt.year
    yearly = kline.groupby('year').apply(
        lambda g: pd.Series({
            '純戰術_%': (g['strat_value'].iloc[-1] / g['strat_value'].iloc[0] - 1) * 100,
            '核心衛星_%': (g['cs_value'].iloc[-1] / g['cs_value'].iloc[0] - 1) * 100,
            'BH_%': (g['bh_value'].iloc[-1] / g['bh_value'].iloc[0] - 1) * 100,
        }),
        include_groups=False,
    )
    print(yearly.round(1).to_string())

    return {
        'strat_total_return': kline['strat_value'].iloc[-1] - 1,
        'cs_total_return': kline['cs_value'].iloc[-1] - 1,
        'bh_total_return': kline['bh_value'].iloc[-1] - 1,
        'strat_cagr': cagr(kline['strat_value'].values, dates),
        'cs_cagr': cagr(kline['cs_value'].values, dates),
        'bh_cagr': cagr(kline['bh_value'].values, dates),
        'strat_mdd': max_drawdown(kline['strat_value'].values),
        'cs_mdd': max_drawdown(kline['cs_value'].values),
        'bh_mdd': max_drawdown(kline['bh_value'].values),
        'strat_sharpe': sharpe(rets['純戰術(0%起跳)']),
        'cs_sharpe': sharpe(rets['核心70%+子彈30%']),
        'bh_sharpe': sharpe(rets['Buy & Hold']),
    }


if __name__ == '__main__':
    kline = run_backtest()
    stats = summarize(kline)
    kline.to_csv('analysis/backtest_daily.csv', index=False)
    print("\n已輸出逐日回測資料: analysis/backtest_daily.csv")
