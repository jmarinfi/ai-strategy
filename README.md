# AI Strategy

AI-powered trading system using ML/DL models for cryptocurrency time series prediction.

## Overview

This system connects to cryptocurrency exchanges, generates predictions using trained models, and sends webhook signals to trading bots.

**Tech Stack:** Python 3.12+ • LightGBM • PyTorch • CCXT • Pydantic

## Quick Start

```bash
# Install dependencies
uv sync

# Train a model
uv run python main.py  # Set MODE='train' in main.py

# Run live trading
uv run python main.py  # Set MODE='live' in main.py
```

## Project Structure

```
src/ai_strategy/
├── data/           # Exchange APIs, WebSockets, preprocessing
├── models/         # ML (sklearn/) and DL (torch/) models
├── strategies/     # Signal generation logic (BaseStrategy, LightGBMStrategy)
├── execution/      # Webhook sender
├── backtesting/    # Backtesting engine
└── utils/          # Config, logging
```

## Key Features

- **Real-time Data**: WebSocket streams from exchanges via CCXT
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, Volume, Momentum
- **Model-Based Predictions**: LightGBM classifier for price movement
- **Strategy Pattern**: Extensible base class for custom strategies
- **Webhook Integration**: Automated signal delivery to trading bots

## Usage

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

## Development

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check --fix .

# Run tests
uv run pytest
```

## Configuration

- **Models**: Store trained models in `models/` directory
- **Data**: Historical data cached in `data/raw/`
- **Configs**: Strategy configs in `configs/` (YAML)

See `.agent/agent.md` for detailed development guidelines.
