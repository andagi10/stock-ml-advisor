"""
Helper para manejar caché de modelos ML
"""
import streamlit as st
from models.predictor import ImprovedStockPredictor
from data.processors.feature_engineering import FeatureEngineer


def load_data_for_ml_cached(ticker):
    """Wrapper para cargar datos con caché"""
    from data.collectors.market_data import MarketDataCollector
    
    collector = MarketDataCollector()
    data = collector.download_ticker(ticker, period='max')
    
    if data is not None:
        engineer = FeatureEngineer()
        data = engineer.add_technical_indicators(data)
    
    return data


def train_or_load_model(ticker, model_type='random_forest', test_size=0.2, 
                        use_cache=True, max_age_days=7, force_retrain=False):
    """
    Entrena un modelo nuevo o carga uno desde caché
    
    Args:
        ticker: Símbolo del ticker
        model_type: Tipo de modelo a usar
        test_size: Proporción de datos para test
        use_cache: Si usar caché
        max_age_days: Edad máxima del modelo en días
        force_retrain: Forzar re-entrenamiento
        
    Returns:
        tuple: (predictor, data_clean, from_cache, metadata)
    """
    # Verificar caché
    cache_path = ImprovedStockPredictor.get_model_cache_path(
        ticker, 
        model_type=model_type,
        cache_dir='models/cache'
    )
    
    should_retrain = force_retrain
    metadata = None
    
    if use_cache and not force_retrain:
        is_valid, cache_metadata = ImprovedStockPredictor.check_cache_validity(
            cache_path, 
            max_age_days=max_age_days
        )
        
        if is_valid:
            # Intentar cargar desde caché
            try:
                st.info(f"""
                ♻️ **Modelo encontrado en caché**  
                📅 Entrenado: {cache_metadata['trained_date'].strftime('%Y-%m-%d %H:%M')}  
                📊 Antigüedad: {cache_metadata['age_days']} día(s)  
                🎯 Modelos: {cache_metadata['n_models']}
                """)
                
                predictor = ImprovedStockPredictor(model_type=model_type)
                metadata = predictor.load_models(cache_path, verbose=False)
                
                # Cargar datos
                st.info("📥 Cargando datos actuales...")
                data_full = load_data_for_ml_cached(ticker)
                data_clean = data_full.dropna()
                
                st.success(f"✅ Modelo cargado desde caché ({len(data_clean):,} muestras)")
                
                return predictor, data_clean, True, metadata
                
            except Exception as e:
                st.warning(f"⚠️ Error cargando caché: {e}. Entrenando nuevo modelo...")
                should_retrain = True
        else:
            if cache_metadata and cache_metadata.get('reason') == 'expired':
                st.warning(f"⚠️ Modelo expirado ({cache_metadata['age_days']} días > {max_age_days}). Re-entrenando...")
            should_retrain = True
    
    # ENTRENAR NUEVO MODELO
    if should_retrain or not use_cache:
        st.info("🔄 Entrenando nuevo modelo desde cero...")
        
        # Cargar datos
        st.info("📥 Descargando histórico completo...")
        data_full = load_data_for_ml_cached(ticker)
        
        if data_full is None or len(data_full) < 50:
            raise ValueError(f"Datos insuficientes: {len(data_full) if data_full else 0} registros")
        
        data_clean = data_full.dropna()
        st.success(f"✅ {len(data_clean):,} muestras listas para entrenamiento")
        
        # Entrenar
        predictor = ImprovedStockPredictor(model_type=model_type)
        
        with st.spinner('🧠 Entrenando modelos multi-horizonte...'):
            trained_count = predictor.train_all_horizons(
                data_clean, 
                test_size=test_size
            )
        
        st.success(f"✅ {trained_count} modelos entrenados")
        
        # Guardar en caché
        if use_cache:
            predictor.save_models(
                filepath=cache_path,
                ticker=ticker,
                data_hash=None
            )
            st.success(f"💾 Guardado en caché: {cache_path}")
        
        return predictor, data_clean, False, None