import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # <--- IMPORTANTE: Para guardar el modelo
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

# ==========================================
# 1. CARGA Y PREPARACIÓN DE DATOS CON INDICADORES TÉCNICOS
# ==========================================
def load_and_prep_data(filepath):
    print(f"Intentando cargar datos desde: {filepath}")
    try:
        df = pd.read_parquet(filepath)
    except FileNotFoundError:
        print(f"\n!!! ERROR: No se encontró {filepath} !!!")
        sys.exit(1)
    
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    print("\n📊 Generando features técnicas...")
    
    # ===== TARGET =====
    df['target'] = df['close'].shift(-1)
    
    # ===== PRECIO - Returns y cambios relativos =====
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # ===== LAGS (Precio pasado) =====
    for lag in [1, 3, 5, 10, 15, 30, 60]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
    
    # ===== MEDIAS MÓVILES (Tendencia) =====
    for period in [9, 21, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        # Precio relativo a MA
        df[f'close_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
    
    # ===== RSI (Momentum) =====
    for period in [14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # ===== MACD (Momentum/Tendencia) =====
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # ===== BANDAS DE BOLLINGER (Volatilidad) =====
    for period in [20, 50]:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df[f'bb_upper_{period}'] = sma + (2 * std)
        df[f'bb_lower_{period}'] = sma - (2 * std)
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
        df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
    
    # ===== ATR (Volatilidad) =====
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(window=14).mean()
    df['atr_21'] = true_range.rolling(window=21).mean()
    
    # ===== STOCHASTIC (Momentum) =====
    for period in [14, 21]:
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        df[f'stoch_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df[f'stoch_{period}_smooth'] = df[f'stoch_{period}'].rolling(window=3).mean()
    
    # ===== OBV (Volumen) =====
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
    
    # ===== VOLUME FEATURES =====
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # ===== PRICE ACTION =====
    # High-Low range
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    # Body (close-open) como % del precio
    df['body_size'] = (df['close'] - df['open']) / df['close']
    
    # ===== MOMENTUM ADICIONAL =====
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['roc_10'] = df['close'].pct_change(periods=10)  # Rate of change
    
    print(f"✅ Features generadas: {len([c for c in df.columns if c != 'target'])} features")
    
    # Limpiar datos
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    print(f"✅ Datos limpios: {len(df)} registros")
    
    return df

# ==========================================
# 2. SCORER PERSONALIZADO (RENTABILIDAD)
# ==========================================
def profit_scorer(estimator, X, y, threshold_pct=0.0001):
    """
    Scorer personalizado que calcula la rentabilidad de la estrategia.
    Optimiza directamente para retornos de trading.
    """
    # Predicciones
    y_pred = estimator.predict(X)
    
    # Crear DataFrame temporal
    df_temp = pd.DataFrame({
        'price': y.values if isinstance(y, pd.Series) else y,
        'pred': y_pred,
    })
    
    # Simular precio actual (shift)
    df_temp['current_price'] = df_temp['price'].shift(1)
    df_temp.dropna(inplace=True)
    
    if len(df_temp) < 10:
        return -999  # Penalización para datos insuficientes
    
    # Señales de trading
    df_temp['long_signal'] = df_temp['pred'] > df_temp['current_price'] * (1 + threshold_pct)
    df_temp['short_signal'] = df_temp['pred'] < df_temp['current_price'] * (1 - threshold_pct)
    
    # Calcular retornos
    df_temp['market_return'] = np.log(df_temp['price'] / df_temp['current_price'])
    df_temp['strategy_return'] = 0.0
    df_temp.loc[df_temp['long_signal'], 'strategy_return'] = df_temp.loc[df_temp['long_signal'], 'market_return']
    df_temp.loc[df_temp['short_signal'], 'strategy_return'] = -df_temp.loc[df_temp['short_signal'], 'market_return']
    
    # Rentabilidad total (retorno acumulado)
    total_return = df_temp['strategy_return'].sum()
    
    return total_return

# ==========================================
# 3. FUNCIÓN DE OPTIMIZACIÓN
# ==========================================
def optimize_model(X, y, model, param_dist, cv, n_iter=10, model_name="Model", use_profit_scorer=False):
    """Optimiza hiperparámetros usando MAE o rentabilidad."""
    print(f"\n--- Optimizando {model_name} ---")
    
    if use_profit_scorer:
        scoring = profit_scorer
        print("  📈 Optimizando para RENTABILIDAD")
    else:
        scoring = 'neg_mean_absolute_error'
        print("  📊 Optimizando para MAE")
    
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    search.fit(X, y)
    
    if use_profit_scorer:
        print(f"  ✅ Mejor rentabilidad: {search.best_score_:.6f}")
    else:
        print(f"  ✅ Mejor MAE: ${-search.best_score_:.2f}")
    
    return search.best_estimator_, search.cv_results_

# ==========================================
# 3. GENERADOR DE META-FEATURES
# ==========================================
def generate_meta_features(X, y, estimators, cv):
    meta_preds = {name: [] for name, _ in estimators}
    meta_y_true = []

    for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train = y.iloc[train_idx]
        meta_y_true.extend(y.iloc[val_idx].values)
        
        for name, estimator in estimators:
            model = clone(estimator)
            model.fit(X_fold_train, y_fold_train)
            preds = model.predict(X_fold_val)
            meta_preds[name].extend(preds)
            
    return meta_preds, np.array(meta_y_true)

# ==========================================
# 4. SIMULADOR DE ESTRATEGIA (BACKTEST)
# ==========================================
def run_simple_backtest(y_true, y_pred, threshold_pct=0.001):
    df_bt = pd.DataFrame({'price': y_true, 'pred': y_pred}, index=y_true.index)
    df_bt['current_price'] = df_bt['price'].shift(1)
    df_bt.dropna(inplace=True)
    
    # Señales
    df_bt['long_signal'] = df_bt['pred'] > df_bt['current_price'] * (1 + threshold_pct)
    df_bt['short_signal'] = df_bt['pred'] < df_bt['current_price'] * (1 - threshold_pct)
    
    # Retornos
    df_bt['market_return'] = np.log(df_bt['price'] / df_bt['current_price'])
    df_bt['strategy_return'] = 0.0
    df_bt.loc[df_bt['long_signal'], 'strategy_return'] = df_bt.loc[df_bt['long_signal'], 'market_return']
    df_bt.loc[df_bt['short_signal'], 'strategy_return'] = -df_bt.loc[df_bt['short_signal'], 'market_return']
    
    # Acumulado
    df_bt['cum_market'] = df_bt['market_return'].cumsum()
    df_bt['cum_strategy'] = df_bt['strategy_return'].cumsum()
    
    return df_bt

# ==========================================
# 5. SISTEMA DE GUARDADO/CARGA
# ==========================================
def save_stacking_system(base_models, meta_model, filename="btc_stacking_system.pkl"):
    """Empaqueta todo el sistema en un diccionario y lo guarda."""
    system_bundle = {
        'base_models': base_models,
        'meta_model': meta_model,
        'version': '1.0',
        'desc': 'Stacking HGB+RF trained on BTC 1m'
    }
    joblib.dump(system_bundle, filename)
    print(f"\n[SISTEMA] Modelo guardado exitosamente en '{filename}'")

def load_and_predict_live(filename, new_data_features):
    """
    Simula cómo se usaría el modelo en vivo.
    new_data_features: DataFrame con una sola fila (la vela actual) y las columnas preparadas.
    """
    print(f"\n[LIVE] Cargando '{filename}' para predicción...")
    bundle = joblib.load(filename)
    
    base_models = bundle['base_models']
    meta_model = bundle['meta_model']
    
    # 1. Obtener opiniones de los modelos base
    preds_base = []
    print("  -> Consultando expertos...")
    for name, model in base_models.items():
        p = model.predict(new_data_features)
        preds_base.append(p)
        print(f"     {name}: ${p[0]:.2f}")
    
    # 2. Preparar input para el jefe
    # (Necesitamos transponer para que tenga forma correcta si es una sola fila)
    X_meta_live = np.column_stack(preds_base)
    
    # 3. Predicción final
    final_pred = meta_model.predict(X_meta_live)[0]
    print(f"  -> PREDICCIÓN FINAL (Stacking): ${final_pred:.2f}")
    return final_pred

# ==========================================
# 6. EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    DATA_PATH = 'data/raw/btc_usdt_15m.parquet'
    df = load_and_prep_data(DATA_PATH)

    features = [col for col in df.columns if col != 'target']
    split_point = int(len(df) * 0.9)

    X_train = df.iloc[:split_point][features]
    y_train = df.iloc[:split_point]['target']
    X_test = df.iloc[split_point:][features]
    y_test = df.iloc[split_point:]['target']

    print(f"\nDatos listos. Train: {len(X_train)} | Test: {len(X_test)}")
    tscv = TimeSeriesSplit(n_splits=5)

    # --- FASE 1: OPTIMIZACIÓN DE HIPERPARÁMETROS ---
    print("\n" + "="*80)
    print("INICIANDO OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("="*80)
    
    # Espacios de búsqueda para HistGradientBoosting (SIMPLIFICADOS)
    hgb_param_dist = {
        'max_depth': [3, 5, 7],  # Reducido (era hasta 20)
        'learning_rate': [0.01, 0.05, 0.1],
        'max_iter': [100, 200, 300],  # Reducido (era hasta 500)
        'min_samples_leaf': [50, 100, 200],  # Aumentado (era desde 10)
        'l2_regularization': [1.0, 5.0, 10.0]  # Aumentado (más regularización)
    }
    
    # Espacios de búsqueda para RandomForest (SIMPLIFICADOS)
    rf_param_dist = {
        'n_estimators': [50, 100, 150],  # Reducido (era hasta 200)
        'max_depth': [5, 7, 10],  # Reducido (era hasta 20 y None)
        'min_samples_leaf': [20, 50, 100],  # Aumentado (era desde 2)
        'max_features': [0.3, 0.5, 0.7]  # Reducido (no usar todas las features)
    }
    
    # Optimizar HGB con MAE (métrica estadística pura)
    best_hgb, hgb_cv_results = optimize_model(
        X_train, y_train,
        HistGradientBoostingRegressor(random_state=42),
        hgb_param_dist,
        tscv,
        n_iter=30,
        model_name="HistGradientBoosting"
        # use_profit_scorer=False por defecto
    )
    
    # Optimizar RandomForest con MAE (métrica estadística pura)
    best_rf, rf_cv_results = optimize_model(
        X_train, y_train,
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_param_dist,
        tscv,
        n_iter=30,
        model_name="RandomForest"
        # use_profit_scorer=False por defecto
    )
    
    print("\n" + "="*80)
    print(f"✅ Mejores parámetros HGB: {best_hgb.get_params()}")
    print(f"✅ Mejores parámetros RF: {best_rf.get_params()}")
    print("="*80)
    
    estimators_opt = [('hgb', best_hgb), ('rf', best_rf)]

    # --- FASE 2: STACKING ---
    print("\n[FASE 1] Entrenando Stacking...")
    meta_preds_dict, y_meta = generate_meta_features(X_train, y_train, estimators_opt, tscv)
    X_meta = np.column_stack([meta_preds_dict['hgb'], meta_preds_dict['rf']])

    meta_model = RidgeCV()
    meta_model.fit(X_meta, y_meta)

    # Re-entrenamiento
    final_models = {}
    for name, model in estimators_opt:
        model.fit(X_train, y_train)
        final_models[name] = model

    # --- FASE 3: EVALUACIÓN ---
    test_preds_hgb = final_models['hgb'].predict(X_test)
    test_preds_rf = final_models['rf'].predict(X_test)
    X_test_meta = np.column_stack([test_preds_hgb, test_preds_rf])
    y_pred = meta_model.predict(X_test_meta)

    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n--- RESULTADOS ---")
    print(f"MAE Final: ${mae:.2f}")

    # --- FASE 4: GRID SEARCH DE THRESHOLD ---
    print("\n" + "="*80)
    print("OPTIMIZACIÓN DE THRESHOLD (Grid Search)")
    print("="*80)
    
    # Thresholds a probar (desde 0.02% hasta 0.5%)
    thresholds_to_test = [0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005]
    
    print(f"\nProbando {len(thresholds_to_test)} thresholds diferentes...")
    
    threshold_results = []
    for threshold in thresholds_to_test:
        df_bt_temp = run_simple_backtest(y_test, y_pred, threshold_pct=threshold)
        
        total_return = df_bt_temp['cum_strategy'].iloc[-1]
        n_trades = (df_bt_temp['long_signal'] | df_bt_temp['short_signal']).sum()
        
        # Sharpe ratio simple
        returns = df_bt_temp['strategy_return']
        sharpe = (returns.mean() / returns.std()) * np.sqrt(35040) if returns.std() > 0 else 0  # 35040 = velas 15m por año
        
        threshold_results.append({
            'threshold': threshold,
            'threshold_pct': f"{threshold*100:.3f}%",
            'return': total_return,
            'return_pct': f"{total_return*100:.2f}%",
            'sharpe': sharpe,
            'n_trades': n_trades
        })
        
        print(f"  Threshold {threshold*100:.3f}%: Retorno={total_return*100:+6.2f}% | Sharpe={sharpe:+6.3f} | Trades={n_trades}")
    
    # Seleccionar mejor threshold basado en rentabilidad
    best_threshold_result = max(threshold_results, key=lambda x: x['return'])
    best_threshold = best_threshold_result['threshold']
    
    print("\n" + "="*80)
    print(f"✅ MEJOR THRESHOLD: {best_threshold*100:.3f}%")
    print(f"   Retorno: {best_threshold_result['return_pct']}")
    print(f"   Sharpe: {best_threshold_result['sharpe']:.3f}")
    print(f"   Trades: {best_threshold_result['n_trades']}")
    print("="*80)

    # --- FASE 5: BACKTEST FINAL CON MEJOR THRESHOLD ---
    print(f"\n[BACKTEST FINAL] Usando threshold óptimo: {best_threshold*100:.3f}%")
    df_bt = run_simple_backtest(y_test, y_pred, threshold_pct=best_threshold)
    
    total_ret = df_bt['cum_strategy'].iloc[-1]
    market_ret = df_bt['cum_market'].iloc[-1]
    
    print(f"Retorno Acumulado Estrategia: {total_ret*100:.2f}%")
    print(f"Retorno Acumulado Mercado (Buy&Hold): {market_ret*100:.2f}%")

    # Guardar gráfico Backtest
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    last_n = 100
    plt.plot(y_test.index[-last_n:], y_test.values[-last_n:], label='Precio Real', color='black', alpha=0.5)
    plt.plot(y_test.index[-last_n:], y_pred[-last_n:], label='Modelo', color='green', linestyle='--')
    plt.title('Predicción vs Realidad (Zoom)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(df_bt.index, df_bt['cum_strategy'], label='Estrategia ML', color='blue')
    plt.plot(df_bt.index, df_bt['cum_market'], label='Mercado (BTC)', color='gray', alpha=0.5)
    plt.title('Curva de Rentabilidad Acumulada (Backtest)')
    plt.ylabel('Retorno Logarítmico')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('resultados_backtest.png')

    # --- FASE 5: GUARDAR PARA PRODUCCIÓN ---
    save_filename = "btc_stacking_production.pkl"
    save_stacking_system(final_models, meta_model, save_filename)
    
    # --- DEMOSTRACIÓN DE USO EN VIVO ---
    # Simulamos que llega un nuevo dato (la última fila del test set)
    print("\n--- SIMULACIÓN DE USO EN VIVO ---")
    sample_live_data = X_test.iloc[[-1]] # Tomamos la última vela conocida
    load_and_predict_live(save_filename, sample_live_data)