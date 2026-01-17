"""Verify completeness of parquet data file."""

from pathlib import Path

import pandas as pd


def verify_parquet(file_path: str | Path) -> None:
    """Verify and analyze a parquet file with OHLCV data.
    
    Args:
        file_path: Path to the parquet file to verify.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"📊 Analyzing: {file_path}")
    print("=" * 80)
    
    # Read the parquet file
    df = pd.read_parquet(file_path)
    
    # Basic statistics
    print(f"\n📈 Basic Statistics:")
    print(f"  Total candles: {len(df):,}")
    print(f"  File size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    if df.empty:
        print("❌ DataFrame is empty!")
        return
    
    # Time range
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    first_candle = df['datetime'].min()
    last_candle = df['datetime'].max()
    
    print(f"\n📅 Time Range:")
    print(f"  First candle: {first_candle}")
    print(f"  Last candle:  {last_candle}")
    print(f"  Duration: {last_candle - first_candle}")
    
    # Expected vs actual candles (for 1m timeframe)
    expected_minutes = int((last_candle - first_candle).total_seconds() / 60) + 1
    print(f"\n🔍 Completeness Check (assuming 1m timeframe):")
    print(f"  Expected candles: {expected_minutes:,}")
    print(f"  Actual candles:   {len(df):,}")
    print(f"  Missing:          {expected_minutes - len(df):,}")
    print(f"  Completeness:     {len(df) / expected_minutes * 100:.2f}%")
    
    # Check for gaps
    print(f"\n🕳️  Gap Analysis:")
    df = df.sort_values('timestamp')
    df['time_diff'] = df['timestamp'].diff()
    
    # For 1m candles, diff should be 60000 ms (1 minute)
    expected_diff = 60_000
    gaps = df[df['time_diff'] > expected_diff]
    
    if len(gaps) > 0:
        print(f"  ⚠️  Found {len(gaps)} gaps:")
        for idx, row in gaps.head(10).iterrows():
            gap_minutes = int(row['time_diff'] / 60_000)
            print(f"    - {row['datetime']}: {gap_minutes} minutes missing")
        if len(gaps) > 10:
            print(f"    ... and {len(gaps) - 10} more gaps")
    else:
        print(f"  ✅ No gaps found!")
    
    # Data quality
    print(f"\n📊 Data Quality:")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Null values: {df.isnull().sum().sum()}")
    print(f"  Duplicates: {df.duplicated(subset=['timestamp']).sum()}")
    
    # Sample data
    print(f"\n📝 Sample Data (first 5 rows):")
    print(df.head())
    
    print("\n" + "=" * 80)
    
    # Overall verdict
    if len(df) >= expected_minutes * 0.95 and len(gaps) == 0:
        print("✅ Data looks complete and healthy!")
    elif len(df) >= expected_minutes * 0.80:
        print("⚠️  Data is mostly complete but has some issues")
    else:
        print("❌ Data appears incomplete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "data/raw/btc_usdt_1m.parquet"
    
    verify_parquet(file_path)
