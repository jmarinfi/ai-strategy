import pandas as pd
import numpy as np

# Load raw OHLCV data
data = pd.read_parquet("data/raw/ada_usdt_15m.parquet")
print(data.head())

# Select features and convert to numpy array (first 1000 rows)
data = data[["open", "high", "low", "volume"]].iloc[:1000, :].copy().to_numpy()
print(f"Data shape: {data.shape}")
print(data[:5, :])

# Create non-overlapping window samples
samples = []
length = 10

for i in range(0, data.shape[0], length):
    samples.append(data[i : i + length])

# Convert samples list to a 3D numpy array
samples = np.array(samples)
print(f"Samples shape: {samples.shape}")
print(samples[:1, :])
