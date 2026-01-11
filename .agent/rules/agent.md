---
trigger: always_on
---

# AI Strategy - LLM Assistant Instructions

## Project Overview

**ai-strategy** is an AI-powered trading strategy system that uses ML/DL models for time series prediction. It connects to cryptocurrency exchanges, generates predictions, and sends webhook signals to trading bots.

---

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Language** | Python 3.12+ |
| **Package Manager** | uv |
| **ML** | scikit-learn, xgboost, lightgbm |
| **Deep Learning** | PyTorch |
| **Exchange Data** | ccxt (REST), websockets (real-time) |
| **Config** | pydantic, PyYAML |
| **HTTP** | httpx, aiohttp |

---

## Project Structure

```
src/ai_strategy/
├── data/           # Exchange APIs, WebSockets, preprocessing
├── models/         # ML (sklearn/) and DL (torch/) models
├── strategies/     # Signal generation logic
├── execution/      # Webhook sender
├── backtesting/    # Backtesting engine
└── utils/          # Config, logging
```

---

## Coding Standards

### General
- **Type hints**: Required on all function signatures
- **Docstrings**: Google-style for public functions/classes
- **Formatting**: `ruff format` (line length 88)
- **Linting**: `ruff check`

### Async
- Use `async/await` for I/O operations (API calls, WebSockets)
- Prefer `asyncio.gather()` for concurrent operations

### Models
- All models inherit from `BaseModel` in `models/base.py`
- Implement `fit(X, y)` and `predict(X)` methods
- PyTorch models: use `torch.nn.Module` + custom wrapper

### Configuration
- Use Pydantic models for validation
- YAML files in `configs/` directory
- Environment variables via `pydantic-settings`

---

## Common Commands

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --dev

# Run training
uv run python scripts/train.py --config configs/model_config.yaml

# Run backtest
uv run python scripts/backtest.py --config configs/backtest_config.yaml

# Run live trading
uv run python scripts/run_live.py --config configs/live_config.yaml

# Run tests
uv run pytest

# Format & lint
uv run ruff format .
uv run ruff check --fix .
```

---

## Key Interfaces

### BaseModel (models/base.py)
```python
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```

### BaseFetcher (data/fetchers/base.py)
```python
class BaseFetcher(ABC):
    @abstractmethod
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int, limit: int
    ) -> pd.DataFrame: ...
```

### BaseStream (data/streams/base.py)
```python
class BaseStream(ABC):
    @abstractmethod
    async def connect(self) -> None: ...
    
    @abstractmethod
    async def subscribe(self, symbols: list[str]) -> None: ...
    
    @abstractmethod
    async def on_message(self, callback: Callable) -> None: ...
```

---

## Workflow Guidelines

1. **Data fetching**: Always handle rate limits and retries
2. **Feature engineering**: Keep preprocessing reproducible (save scalers)
3. **Model training**: Log hyperparameters and metrics
4. **Predictions**: Validate output ranges before generating signals
5. **Webhooks**: Implement retry logic with exponential backoff

---

## Error Handling

- Use custom exceptions in `utils/exceptions.py`
- Log errors with `loguru` + context
- Never swallow exceptions silently
- API errors: implement circuit breaker pattern

---

## Testing

- Unit tests in `tests/` mirroring `src/` structure
- Use `pytest-asyncio` for async tests
- Mock external APIs with `pytest-httpx`
- Target coverage: >80%
