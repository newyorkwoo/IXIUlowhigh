import yfinance as yf
import pandas as pd

df = yf.Ticker('^IXIC').history(start='2000-01-01')
if hasattr(df.index, 'tz') and df.index.tz is not None:
    df.index = df.index.tz_localize(None)

# Calculate 7-day rolling return (Drop Speed over 7 days)
df['ROC_7'] = df['Close'].pct_change(periods=7) * 100

# Find extremely fast drops
print("=== Top 15 Fastest 7-Day Drops ===")
worst_drops = df.sort_values('ROC_7').head(15)
print(worst_drops[['Close', 'ROC_7']])

# Let's check the rolling 7-day drop on our known flash crashes:
print("\n=== Specific Dates ===")
dates = ['2008-10-10', '2008-10-24', '2008-11-20', '2015-08-24', '2020-03-12', '2020-03-16', '2020-03-20', '2024-08-05']
for d in dates:
    try:
        # Get the row closest to this date or exact
        idx = df.index.get_indexer([pd.to_datetime(d)], method='nearest')[0]
        row = df.iloc[idx]
        print(f"{row.name.strftime('%Y-%m-%d')} | 7-day drop: {row['ROC_7']:.2f}%")
    except Exception as e:
        pass
