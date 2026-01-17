# LightGBM Strategy Implementation

This directory contains the LightGBM-based trading strategy implementation.

## Architecture

The implementation follows a clean separation of concerns:

### 1. **Model Layer** (`src/ai_strategy/models/sklearn/`)
- `LightGBMModel`: Encapsulates all model-related logic
  - Model loading from disk
  - Feature preparation
  - Probability predictions
  - Model metadata (config, metrics, feature columns)

### 2. **Strategy Layer** (`src/ai_strategy/strategies/`)
- `BaseStrategy`: Abstract base class for all strategies
  - Defines the common interface (`on_candle`, `generate_signal`)
  - Handles webhook sending
  - Manages position opening/closing
  
- `LightGBMStrategy`: Concrete implementation using LightGBM
  - Processes candlestick data
  - Fetches historical data for indicators
  - Calculates technical indicators
  - Uses `LightGBMModel` for predictions
  - Generates trading signals based on probability thresholds

### 3. **Models** (`src/ai_strategy/models/`)
- `Candle`: OHLCV candlestick data model
- `Signal`: Trading signal model (BUY/SELL/HOLD)
- `WebhookRequest`: Base for webhook models

## Data Flow

```
1. Streamer receives OHLCV data
   ↓
2. Strategy detects complete candle
   ↓
3. Fetch historical data (60 candles)
   ↓
4. Calculate technical indicators (RSI, MACD, BB, ATR, Volume, Momentum)
   ↓
5. LightGBMModel.predict(features) → {prob_up, prob_down}
   ↓
6. generate_signal(candle, prediction) → Signal
   ↓
7. If actionable: send_webhook() → Trading bot
```

## Technical Indicators

The strategy calculates the following indicators:

- **RSI** (14, 21 periods)
- **MACD** (12, 26, 9)
- **Bollinger Bands** (20 period, 2 std)
- **ATR** (14 period)
- **Volume** (SMA 20, ratio)
- **Momentum** (5, 10 periods)

## Usage Example

```python
from src.ai_strategy.data import CCXTStreamer
from src.ai_strategy.strategies import LightGBMStrategy

async with CCXTStreamer("bitget", ["BTC/USDT"]) as streamer:
    strategy = LightGBMStrategy(
        symbols=["BTC/USDT"],
        streamer=streamer,
        webhook_url="http://localhost:7503/trade_signal",
        bot_uuid="your-bot-uuid",
        model_path="models/lightgbm_btc_usdt_5m_h6.pkl",
        timeframe="5m",
        prob_threshold=0.52,
    )
    
    await strategy.run()
```

## Configuration

Key parameters:

- `prob_threshold`: Minimum probability to generate BUY/SELL signal (default: 0.52)
- `historical_candles`: Number of candles to fetch for indicators (default: 60)
- `timeframe`: Candle timeframe (e.g., '5m', '1h')
- `exchange`: Exchange name (default: 'bitget')

## Model Requirements

The model file (`.pkl`) must contain:

```python
{
    'model': <trained LightGBM classifier>,
    'feature_cols': <list of feature names>,
    'config': {
        'symbol': str,
        'timeframe': str,
        'horizon': int,
        'min_movement': float,
        ...
    },
    'metrics': {
        'accuracy': float,
        'f1_score': float,
        ...
    }
}
```

Train a model using `main.py` with `MODE='train'`.
