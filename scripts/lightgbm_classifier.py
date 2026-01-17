"""LightGBM Binary Classification para Trading - BTC/USDT 15m"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# ==========================================
# CONFIGURACIÓN
# ==========================================
HORIZON_CANDLES = 16  # 16 velas x 15m = 4 horas
MIN_MOVEMENT = 0.0  # 0% = cualquier movimiento (puedes ajustar a 0.002 para >=0.2%)

# ==========================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==========================================
def load_and_prep_data(filepath, horizon=HORIZON_CANDLES, min_movement=MIN_MOVEMENT):
    """Carga datos y genera features técnicas + target de clasificación."""
    print(f"📂 Cargando datos desde: {filepath}")
    df = pd.read_parquet(filepath)
    
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"\n📊 Generando features técnicas...")
    print(f"🎯 Horizonte de predicción: {horizon} velas ({horizon * 15} minutos = {horizon * 15 / 60:.1f} horas)")
    
    # ===== TARGET: Clasificación binaria con horizonte largo =====
    df['future_price'] = df['close'].shift(-horizon)
    df['future_return'] = (df['future_price'] - df['close']) / df['close']
    
    if min_movement > 0:
        # Solo predecir movimientos significativos
        df['target'] = (df['future_return'] > min_movement).astype(int)
        print(f"📈 Movimiento mínimo: {min_movement*100:.2f}%")
    else:
        # Predecir cualquier subida
        df['target'] = (df['future_return'] > 0).astype(int)
        print(f"📈 Predicción: Sube vs Baja (cualquier movimiento)")
    
    # ===== PRECIO - Returns =====
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # ===== LAGS =====
    for lag in [1, 3, 5, 10]:
        df[f'return_lag_{lag}'] = df['returns'].shift(lag)
    
    # ===== MEDIAS MÓVILES =====
    for period in [9, 21, 50]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
    
    # ===== RSI =====
    for period in [14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1e-10)  # Evitar división por cero
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # ===== MACD =====
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ===== BOLLINGER BANDS =====
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma_20 + (2 * std_20)
    df['bb_lower'] = sma_20 - (2 * std_20)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ===== ATR (Volatility) =====
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    
    # ===== VOLUME =====
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # ===== MOMENTUM =====
    df['momentum_5'] = df['close'].pct_change(periods=5)
    df['momentum_10'] = df['close'].pct_change(periods=10)
    
    print(f"✅ Total features: {len([c for c in df.columns if c not in ['target', 'future_return']])}")
    
    # Limpiar
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    print(f"✅ Datos limpios: {len(df):,} registros")
    print(f"✅ Distribución target: {df['target'].value_counts().to_dict()}")
    
    return df


# ==========================================
# 2. BACKTEST CON PROBABILIDADES
# ==========================================
def run_classification_backtest(y_true, y_proba, threshold=0.5):
    """
    Backtest usando probabilidades del clasificador.
    
    Args:
        y_true: Precios reales (Series con index datetime)
        y_proba: Probabilidades de subida (array)
        threshold: Umbral para generar señal (default 0.5)
    """
    df_bt = pd.DataFrame({
        'price': y_true.values,
        'prob_up': y_proba
    }, index=y_true.index)
    
    df_bt['current_price'] = df_bt['price'].shift(1)
    df_bt.dropna(inplace=True)
    
    # Señales basadas en probabilidad
    df_bt['long_signal'] = df_bt['prob_up'] > threshold
    df_bt['short_signal'] = df_bt['prob_up'] < (1 - threshold)
    
    # Retornos
    df_bt['market_return'] = np.log(df_bt['price'] / df_bt['current_price'])
    df_bt['strategy_return'] = 0.0
    
    # Long: gana si sube
    df_bt.loc[df_bt['long_signal'], 'strategy_return'] = df_bt.loc[df_bt['long_signal'], 'market_return']
    # Short: gana si baja
    df_bt.loc[df_bt['short_signal'], 'strategy_return'] = -df_bt.loc[df_bt['short_signal'], 'market_return']
    
    # Acumulados
    df_bt['cum_market'] = df_bt['market_return'].cumsum()
    df_bt['cum_strategy'] = df_bt['strategy_return'].cumsum()
    
    return df_bt


# ==========================================
# 3. OPTIMIZACIÓN DE THRESHOLD
# ==========================================
def optimize_threshold(y_test_prices, y_proba):
    """Encuentra el mejor threshold de probabilidad."""
    print("\n" + "="*80)
    print("OPTIMIZACIÓN DE THRESHOLD")
    print("="*80)
    
    thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
    results = []
    
    for thresh in thresholds:
        df_bt = run_classification_backtest(y_test_prices, y_proba, threshold=thresh)
        
        total_return = df_bt['cum_strategy'].iloc[-1]
        n_trades = (df_bt['long_signal'] | df_bt['short_signal']).sum()
        
        # Sharpe
        returns = df_bt['strategy_return']
        sharpe = (returns.mean() / returns.std()) * np.sqrt(35040) if returns.std() > 0 else 0
        
        results.append({
            'threshold': thresh,
            'return': total_return,
            'sharpe': sharpe,
            'n_trades': n_trades
        })
        
        print(f"  Threshold {thresh:.2f}: Return={total_return*100:+6.2f}% | Sharpe={sharpe:+6.3f} | Trades={n_trades}")
    
    # Mejor threshold
    best = max(results, key=lambda x: x['return'])
    print(f"\n✅ MEJOR THRESHOLD: {best['threshold']:.2f}")
    print(f"   Return: {best['return']*100:+.2f}%")
    print(f"   Sharpe: {best['sharpe']:.3f}")
    print("="*80)
    
    return best['threshold']


# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    # Cargar datos con horizonte de 4 horas
    DATA_PATH = 'data/raw/btc_usdt_15m.parquet'
    df = load_and_prep_data(DATA_PATH, horizon=HORIZON_CANDLES, min_movement=MIN_MOVEMENT)
    
    # Preparar features y target
    feature_cols = [c for c in df.columns if c not in ['target', 'future_return', 'future_price']]
    X = df[feature_cols]
    y = df['target']
    
    # Split temporal
    split_point = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    print(f"\n📊 Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # ===== ENTRENAR LIGHTGBM =====
    print("\n" + "="*80)
    print("ENTRENANDO LIGHTGBM CLASIFICADOR")
    print("="*80)
    
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='binary_logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    # ===== PREDICCIONES =====
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidad de clase 1 (sube)
    
    # ===== MÉTRICAS DE CLASIFICACIÓN =====
    print("\n" + "="*80)
    print("MÉTRICAS DE CLASIFICACIÓN")
    print("="*80)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=['Baja', 'Sube']))
    
    # ===== OPTIMIZAR THRESHOLD =====
    y_test_prices = df['close'].iloc[split_point:]
    best_threshold = optimize_threshold(y_test_prices, y_proba)
    
    # ===== BACKTEST FINAL =====
    print("\n" + "="*80)
    print("BACKTEST FINAL")
    print("="*80)
    
    df_bt = run_classification_backtest(y_test_prices, y_proba, threshold=best_threshold)
    
    final_return = df_bt['cum_strategy'].iloc[-1]
    market_return = df_bt['cum_market'].iloc[-1]
    n_trades = (df_bt['long_signal'] | df_bt['short_signal']).sum()
    
    print(f"Retorno Estrategia: {final_return*100:+.2f}%")
    print(f"Retorno Mercado:    {market_return*100:+.2f}%")
    print(f"Número de trades:   {n_trades}")
    
    # ===== GRÁFICOS =====
    plt.figure(figsize=(14, 10))
    
    # Gráfico 1: Probabilidades
    plt.subplot(3, 1, 1)
    plt.plot(df_bt.index, df_bt['prob_up'], label='P(Sube)', color='green', alpha=0.7)
    plt.axhline(y=best_threshold, color='red', linestyle='--', label=f'Threshold={best_threshold}')
    plt.axhline(y=1-best_threshold, color='blue', linestyle='--')
    plt.title('Probabilidades del Modelo')
    plt.ylabel('Probabilidad')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 2: Precio y señales
    plt.subplot(3, 1, 2)
    plt.plot(df_bt.index, df_bt['price'], label='Precio BTC', color='black', alpha=0.5)
    # Señales long
    long_signals = df_bt[df_bt['long_signal']]
    plt.scatter(long_signals.index, long_signals['price'], color='green', marker='^', s=50, alpha=0.5, label='Long')
    # Señales short
    short_signals = df_bt[df_bt['short_signal']]
    plt.scatter(short_signals.index, short_signals['price'], color='red', marker='v', s=50, alpha=0.5, label='Short')
    plt.title('Señales de Trading')
    plt.ylabel('Precio ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 3: Rentabilidad acumulada
    plt.subplot(3, 1, 3)
    plt.plot(df_bt.index, df_bt['cum_strategy'], label='Estrategia ML', color='blue', linewidth=2)
    plt.plot(df_bt.index, df_bt['cum_market'], label='Buy & Hold', color='gray', alpha=0.7)
    plt.title('Rentabilidad Acumulada')
    plt.ylabel('Retorno Logarítmico')
    plt.xlabel('Tiempo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lightgbm_backtest_results.png', dpi=150)
    print(f"\n📊 Gráficos guardados en: lightgbm_backtest_results.png")
    
    # ===== GUARDAR MODELO =====
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'best_threshold': best_threshold,
        'metrics': {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'final_return': final_return
        }
    }
    
    joblib.dump(model_data, 'lightgbm_btc_classifier.pkl')
    print(f"✅ Modelo guardado en: lightgbm_btc_classifier.pkl")
