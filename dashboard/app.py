"""
Dashboard interactivo para Stock ML Advisor
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path
import yaml
import os


# Añadir raíz al path
sys.path.append(str(Path(__file__).parent.parent))

# Añadir utils al path para importar model_cache_helper
utils_path = Path(__file__).parent.parent / 'utils'
sys.path.insert(0, str(utils_path))

from data.collectors.market_data import MarketDataCollector
from data.processors.feature_engineering import FeatureEngineer
from models.predictor import ImprovedStockPredictor

# Importar helper de caché
try:
    from model_cache_helper import train_or_load_model
    CACHE_AVAILABLE = True
    print("✅ Sistema de caché cargado correctamente")
except ImportError as e:
    CACHE_AVAILABLE = False
    print(f"⚠️ Advertencia: model_cache_helper no disponible: {e}")
    print(f"   Buscando en: {utils_path}")
    print("   Usando funcionalidad de caché integrada...")
    
    # Funcionalidad de caché integrada directamente
    def train_or_load_model(ticker, model_type='random_forest', test_size=0.2, 
                            use_cache=True, max_age_days=7, force_retrain=False):
        """Versión inline del helper de caché"""
        from datetime import datetime
        import pickle
        import os  # IMPORTANTE: importar os aquí
        
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
                try:
                    st.info(f"""
                    ♻️ **Cargando modelo desde caché**  
                    📅 Entrenado: {cache_metadata['trained_date'].strftime('%Y-%m-%d %H:%M')}  
                    📊 Antigüedad: {cache_metadata['age_days']} día(s)
                    """)
                    
                    predictor = ImprovedStockPredictor(model_type=model_type)
                    metadata = predictor.load_models(cache_path, verbose=False)
                    
                    # Cargar datos
                    st.info("📥 Cargando datos actuales...")
                    collector = MarketDataCollector()
                    data_full = collector.download_ticker(ticker, period='max')
                    engineer = FeatureEngineer()
                    data_featured = engineer.add_technical_indicators(data_full)
                    data_clean = data_featured.dropna()
                    
                    st.success(f"✅ Modelo cargado desde caché ({len(data_clean):,} muestras)")
                    return predictor, data_clean, True, metadata
                    
                except Exception as e:
                    st.warning(f"⚠️ Error cargando caché: {e}. Entrenando nuevo modelo...")
                    should_retrain = True
            else:
                should_retrain = True
        
        # ENTRENAR NUEVO MODELO
        st.info("🔄 Entrenando nuevo modelo desde cero...")
        
        # Cargar datos
        st.info("📥 Descargando histórico completo...")
        collector = MarketDataCollector()
        data_full = collector.download_ticker(ticker, period='max')
        
        if data_full is None or len(data_full) < 50:
            raise ValueError(f"Datos insuficientes: {len(data_full) if data_full else 0} registros")
        
        engineer = FeatureEngineer()
        data_featured = engineer.add_technical_indicators(data_full)
        data_clean = data_featured.dropna()
        st.success(f"✅ {len(data_clean):,} muestras listas para entrenamiento")
        
        # Entrenar
        predictor = ImprovedStockPredictor(model_type=model_type)
        
        with st.spinner('🧠 Entrenando modelos multi-horizonte...'):
            trained_count = predictor.train_all_horizons(data_clean, test_size=test_size)
        
        st.success(f"✅ {trained_count} modelos entrenados")
        
        # Guardar SIEMPRE en caché automáticamente
        try:
            # Asegurar que existe el directorio
            cache_dir = os.path.dirname(cache_path)
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
                st.info(f"📁 Carpeta de caché creada: {cache_dir}")
            
            # Guardar el modelo
            predictor.save_models(filepath=cache_path, ticker=ticker, data_hash=None)
            st.success(f"💾 Modelo guardado automáticamente en caché")
            st.info(f"📍 Ruta: {cache_path}")
        except Exception as e:
            st.error(f"❌ Error guardando en caché: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        
        return predictor, data_clean, False, None
    
    CACHE_AVAILABLE = True  # Activar porque tenemos la versión inline
    print("✅ Usando sistema de caché integrado (inline)")


# Configuración de la página
st.set_page_config(
    page_title="Stock ML Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_data(ticker, period):
    """Carga y procesa datos para visualización"""
    collector = MarketDataCollector()
    data = collector.download_ticker(ticker, period=period)
    
    if data is not None:
        engineer = FeatureEngineer()
        data = engineer.add_technical_indicators(data)
        data = engineer.create_target_variable(data, horizon=5)
    
    return data


def load_data_for_ml(ticker):
    """Carga datos históricos completos para entrenamiento ML"""
    collector = MarketDataCollector()
    # Siempre descargar el máximo histórico disponible para ML
    data = collector.download_ticker(ticker, period='max')
    
    if data is not None:
        engineer = FeatureEngineer()
        data = engineer.add_technical_indicators(data)
        # No crear target_variable aquí, lo hará el predictor
    
    return data


def plot_price_chart(data, ticker):
    """Gráfico de precios con indicadores"""
    fig = go.Figure()
    
    # Precio de cierre
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['close'],
        name='Precio',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # SMA 20
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['sma_20'],
        name='SMA 20',
        line=dict(color='orange', width=1, dash='dash')
    ))
    
    # SMA 50
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['sma_50'],
        name='SMA 50',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['bb_upper'],
        name='BB Superior',
        line=dict(color='gray', width=0.5),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['bb_lower'],
        name='BB Inferior',
        line=dict(color='gray', width=0.5),
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.1)',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f'📈 {ticker} - Análisis Técnico',
        xaxis_title='Fecha',
        yaxis_title='Precio',
        hovermode='x unified',
        height=500
    )
    
    return fig


def plot_indicators(data):
    """Gráficos de indicadores técnicos"""
    # RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=data.index,
        y=data['rsi'],
        name='RSI',
        line=dict(color='purple', width=2)
    ))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecompra")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobreventa")
    fig_rsi.update_layout(
        title='RSI (Relative Strength Index)',
        yaxis_title='RSI',
        height=300
    )
    
    # MACD
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(
        x=data.index,
        y=data['macd'],
        name='MACD',
        line=dict(color='blue', width=2)
    ))
    fig_macd.add_trace(go.Scatter(
        x=data.index,
        y=data['macd_signal'],
        name='Señal',
        line=dict(color='red', width=2)
    ))
    fig_macd.add_trace(go.Bar(
        x=data.index,
        y=data['macd_diff'],
        name='Histograma',
        marker_color='gray'
    ))
    fig_macd.update_layout(
        title='MACD',
        yaxis_title='Valor',
        height=300
    )
    
    return fig_rsi, fig_macd


def plot_returns_distribution(data):
    """Distribución de retornos"""
    fig = px.histogram(
        data.dropna(),
        x='returns',
        nbins=50,
        title='Distribución de Retornos Diarios',
        labels={'returns': 'Retorno Diario'}
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    return fig


def main():
    """Aplicación principal"""
    
    # Título
    st.title("📈 Stock ML Advisor Dashboard")
    st.markdown("### Sistema de Análisis y Predicción de Mercados")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuración")
    
    # Selección de mercado
    market_options = {
        'S&P 500': '^GSPC',
        'IBEX 35': '^IBEX',
        'MSCI World': 'URTH',
        'Santander': 'SAN.MC',
        'Iberdrola': 'IBE.MC',
        'Inditex': 'ITX.MC',
        'EuroStoxx 50': '^STOXX50E'
    }
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Ruta correcta al settings.yaml en config/
    yaml_path = os.path.join(current_dir, "..", "config", "settings.yaml")

    # Verificar que existe el archivo
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"No se encontró settings.yaml en: {yaml_path}")

    # Cargar YAML
    with open(yaml_path, 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)

    # Tomar todos los mercados de todas las secciones y ponerlos en un diccionario
    market_options = {}

    for region, markets in settings['markets'].items():
        for market in markets:
            # Usar el formato "Nombre (Región)" para mejor organización
            display_name = f"{market['name']} ({region.upper()})"
            market_options[display_name] = {
                'ticker': market['ticker'],
                'type': market['type'],
                'region': region
            }
    
    selected_market = st.sidebar.selectbox(
        'Selecciona un mercado:',
        options=list(market_options.keys())
    )
    
    market_info = market_options[selected_market]
    ticker = market_info['ticker']
    
    # Obtener períodos desde configuración
    period_options = settings['data']['periods']['visualization']
    period_labels = [p['label'] for p in period_options]
    period_values = {p['label']: p['value'] for p in period_options}
    
    # Obtener índice del período por defecto
    default_period = settings['data']['periods']['default_visualization']
    try:
        default_idx = [p['value'] for p in period_options].index(default_period)
    except ValueError:
        default_idx = 4  # Default a "1 Año" si no se encuentra
    
    # Período de visualización
    selected_period_label = st.sidebar.selectbox(
        'Período de visualización:',
        options=period_labels,
        index=default_idx
    )
    
    period = period_values[selected_period_label]
    
    # Configuración de auto-actualización ML
    st.sidebar.divider()
    auto_update_ml = st.sidebar.checkbox(
        "🔄 Auto-actualizar Predicciones ML",
        value=True,
        help="Al cargar datos, actualiza automáticamente las predicciones ML si hay modelo en caché"
    )
    
    # Mostrar información del mercado
    st.sidebar.info(f"""
    **Tipo**: {market_info['type'].upper()}  
    **Ticker**: `{ticker}`  
    **Región**: {market_info['region'].title()}
    """)
    
    # Mostrar estado del caché para este ticker
    if CACHE_AVAILABLE:
        cache_dir = 'models/cache'
        if os.path.exists(cache_dir):
            # Buscar modelos para este ticker
            safe_ticker = ticker.replace('^', '').replace('.', '_')
            ticker_caches = [f for f in os.listdir(cache_dir) if f.startswith(safe_ticker) and f.endswith('.pkl')]
            
            if ticker_caches:
                # Obtener info del más reciente
                latest_cache = max(ticker_caches, key=lambda f: os.path.getmtime(os.path.join(cache_dir, f)))
                cache_path = os.path.join(cache_dir, latest_cache)
                mtime = os.path.getmtime(cache_path)
                from datetime import datetime
                cache_date = datetime.fromtimestamp(mtime)
                age_days = (datetime.now() - cache_date).days
                
                st.sidebar.success(f"""
                **💾 Modelo en Caché**  
                📅 {cache_date.strftime('%Y-%m-%d')}  
                ⏰ {age_days} día(s)
                """)
            else:
                st.sidebar.warning("**📝 Sin caché**")
    
    # Botón de carga
    if st.sidebar.button('🔄 Cargar Datos', type='primary'):
        with st.spinner(f'Descargando datos de {selected_market}...'):
            data = load_data(ticker, period)
            st.session_state['data'] = data
            st.session_state['ticker'] = ticker
            st.session_state['market_name'] = selected_market.split(' (')[0]  # Solo el nombre sin la región
            
            # Auto-actualizar predicciones ML si está activado y existe modelo en caché
            if CACHE_AVAILABLE and auto_update_ml:
                cache_path = ImprovedStockPredictor.get_model_cache_path(
                    ticker,
                    model_type='random_forest',  # Por defecto
                    cache_dir='models/cache'
                )
                
                is_valid, cache_metadata = ImprovedStockPredictor.check_cache_validity(
                    cache_path,
                    max_age_days=7
                )
                
                if is_valid:
                    try:
                        with st.spinner('🔄 Actualizando predicciones ML desde caché...'):
                            # Cargar modelo desde caché
                            predictor = ImprovedStockPredictor(model_type='random_forest')
                            predictor.load_models(cache_path, verbose=False)
                            
                            # Cargar datos completos para ML
                            data_full = load_data_for_ml(ticker)
                            engineer = FeatureEngineer()
                            data_featured = engineer.add_technical_indicators(data_full)
                            data_clean = data_featured.dropna()
                            
                            # Guardar en session_state
                            st.session_state['predictor'] = predictor
                            st.session_state['data_clean'] = data_clean
                            
                            # Generar predicciones automáticamente
                            predictions_df = predictor.predict_investment_horizons(
                                data_clean,
                                show_details=False
                            )
                            st.session_state['predictions_df'] = predictions_df
                            
                            st.sidebar.success("✅ Predicciones ML actualizadas automáticamente")
                    except Exception as e:
                        st.sidebar.warning(f"⚠️ No se pudieron actualizar predicciones: {e}")
            elif CACHE_AVAILABLE and not auto_update_ml:
                st.sidebar.info("ℹ️ Auto-actualización ML desactivada. Ve a la pestaña 'Predicciones ML' para actualizar manualmente.")
    
    # Gestión global de caché en sidebar
    if CACHE_AVAILABLE:
        st.sidebar.divider()
        with st.sidebar.expander("💾 Gestión de Caché", expanded=False):
            cache_dir = 'models/cache'
            if os.path.exists(cache_dir):
                cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
                
                st.write(f"**Modelos almacenados: {len(cache_files)}**")
                
                if cache_files:
                    from datetime import datetime
                    # Mostrar resumen por ticker
                    ticker_groups = {}
                    for filename in cache_files:
                        ticker_name = filename.split('_')[0]
                        if ticker_name not in ticker_groups:
                            ticker_groups[ticker_name] = []
                        ticker_groups[ticker_name].append(filename)
                    
                    for ticker_name, files in sorted(ticker_groups.items()):
                        st.text(f"📊 {ticker_name}: {len(files)} modelo(s)")
                    
                    # Botón para limpiar todo el caché
                    if st.button("🗑️ Limpiar Todo Caché", key="clear_all_cache"):
                        for filename in cache_files:
                            try:
                                os.remove(os.path.join(cache_dir, filename))
                            except:
                                pass
                        st.success("✅ Caché limpiado")
                        st.rerun()
                else:
                    st.info("No hay modelos en caché")
            else:
                st.warning("Carpeta de caché no existe")
                if st.button("📁 Crear carpeta caché"):
                    os.makedirs(cache_dir, exist_ok=True)
                    st.success("✅ Carpeta creada")
    
    # Mostrar datos si existen
    if 'data' in st.session_state and st.session_state['data'] is not None:
        data = st.session_state['data']
        ticker = st.session_state['ticker']
        market_name = st.session_state['market_name']
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2]
        change = ((current_price - prev_price) / prev_price) * 100
        
        col1.metric(
            "Precio Actual",
            f"${current_price:.2f}",
            f"{change:+.2f}%"
        )
        
        col2.metric(
            "RSI",
            f"{data['rsi'].iloc[-1]:.1f}",
            "Sobrecompra" if data['rsi'].iloc[-1] > 70 else "Sobreventa" if data['rsi'].iloc[-1] < 30 else "Normal"
        )
        
        col3.metric(
            "Volatilidad (20d)",
            f"{data['volatility'].iloc[-1]*100:.2f}%"
        )
        
        returns_total = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
        col4.metric(
            "Retorno Total",
            f"{returns_total:+.2f}%"
        )
        
        # Tabs para diferentes vistas
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Gráfico Principal",
            "📉 Indicadores Técnicos",
            "📈 Análisis Estadístico",
            "🔮 Predicciones ML",
            "🔢 Datos Crudos"
        ])
        
        with tab1:
            st.plotly_chart(
                plot_price_chart(data, market_name),
                use_container_width=True
            )
        
        with tab2:
            fig_rsi, fig_macd = plot_indicators(data)
            st.plotly_chart(fig_rsi, use_container_width=True)
            st.plotly_chart(fig_macd, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    plot_returns_distribution(data),
                    use_container_width=True
                )
            
            with col2:
                st.subheader("Estadísticas")
                stats_df = pd.DataFrame({
                    'Métrica': [
                        'Retorno Promedio Diario',
                        'Volatilidad Diaria',
                        'Sharpe Ratio (aprox)',
                        'Máximo Drawdown',
                        'Días Positivos',
                        'Días Negativos'
                    ],
                    'Valor': [
                        f"{data['returns'].mean()*100:.3f}%",
                        f"{data['returns'].std()*100:.3f}%",
                        f"{(data['returns'].mean() / data['returns'].std()):.2f}",
                        f"{((data['close'] / data['close'].cummax() - 1).min()*100):.2f}%",
                        f"{(data['returns'] > 0).sum()}",
                        f"{(data['returns'] < 0).sum()}"
                    ]
                })
                st.table(stats_df)
        with tab4:
            show_predictions_tab(data)
        with tab5:
            st.subheader("Datos Históricos")
            st.dataframe(
                data[['close', 'volume', 'sma_20', 'rsi', 'macd', 'returns']].tail(50),
                use_container_width=True
            )
            
            # Botón de descarga
            csv = data.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name=f'{ticker}_data.csv',
                mime='text/csv'
            )
    
    else:
        st.info('👈 Selecciona un mercado y presiona "Cargar Datos" para empezar')
        
        # Imagen o descripción del proyecto
        st.markdown("""
        ## 🎯 Características del Sistema
        
        - **Análisis Multi-Mercado**: S&P500, IBEX35, MSCI World y más
        - **Indicadores Técnicos**: RSI, MACD, Bollinger Bands, SMA
        - **Machine Learning**: Modelos LSTM, Random Forest, XGBoost
        - **Aprendizaje Continuo**: El sistema aprende de sus aciertos y errores
        - **Backtesting**: Prueba estrategias con datos históricos
        
        ## 📊 Próximos Pasos
        
        1. Selecciona un mercado en el panel izquierdo
        2. Elige el período de análisis
        3. Presiona "Cargar Datos"
        4. Explora las diferentes pestañas
        """)

def show_predictions_tab(data):
    """Muestra predicciones del modelo mejorado multi-horizonte"""
    st.subheader("🔮 Predicciones Multi-Horizonte con Machine Learning")
    
    try:
        from models.predictor import ImprovedStockPredictor
        from data.processors.feature_engineering import FeatureEngineer
        
        st.markdown("""
        Este sistema entrena modelos separados para diferentes horizontes de inversión:
        - **Corto plazo**: 1 día, 1 semana, 1 mes
        - **Medio plazo**: 3 meses, 6 meses, 1 año
        - **Largo plazo**: 3 años, 5 años, 10 años, 15 años, 20 años
        
        ℹ️ **Nota**: El sistema usa el máximo histórico disponible para entrenar, independiente del período seleccionado para visualización.
        """)
        
        # Mostrar estado del caché si está disponible
        if CACHE_AVAILABLE:
            ticker = st.session_state.get('ticker', '^GSPC')
            cache_path = ImprovedStockPredictor.get_model_cache_path(
                ticker, 
                model_type='random_forest',
                cache_dir='models/cache'
            )
            
            if os.path.exists(cache_path):
                mtime = os.path.getmtime(cache_path)
                from datetime import datetime
                cache_date = datetime.fromtimestamp(mtime)
                age_days = (datetime.now() - cache_date).days
                
                st.success(f"""
                ✅ **Modelo en caché disponible**  
                📅 Última actualización: {cache_date.strftime('%Y-%m-%d %H:%M')}  
                ⏰ Antigüedad: {age_days} día(s)
                """)
            else:
                st.info("📝 No hay modelo en caché para este ticker. Se entrenará desde cero.")
        
        # Opciones de configuración
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                'Tipo de modelo:',
                options=['random_forest', 'gradient_boosting'],
                index=0
            )
        
        with col2:
            test_size = st.slider(
                'Proporción de datos para test:',
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05
            )
        
        if st.button('🧠 Entrenar Modelos Multi-Horizonte', type='primary'):
            # Obtener el ticker actual
            ticker = st.session_state.get('ticker', '^GSPC')
            
            if CACHE_AVAILABLE:
                # ==================== MODO CON CACHÉ ====================
                
                # Verificar caché existente
                cache_path = ImprovedStockPredictor.get_model_cache_path(
                    ticker, 
                    model_type=model_type,
                    cache_dir='models/cache'
                )
                
                # Configuración de caché
                max_age_days = 7  # Puedes leer esto desde settings.yaml
                
                # Verificar si existe caché válido
                is_valid, cache_metadata = ImprovedStockPredictor.check_cache_validity(
                    cache_path, 
                    max_age_days=max_age_days
                )
                
                # Mostrar estado del caché
                if is_valid:
                    st.info(f"""
                    ♻️ **Modelo encontrado en caché**  
                    📅 Entrenado: {cache_metadata['trained_date'].strftime('%Y-%m-%d %H:%M')}  
                    📊 Antigüedad: {cache_metadata['age_days']} día(s)  
                    🎯 Modelos: {cache_metadata['n_models']}
                    """)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        use_cached = st.checkbox(
                            "✅ Usar modelo en caché (más rápido)", 
                            value=True,
                            help="Desmarcar para re-entrenar desde cero"
                        )
                    with col2:
                        if st.button("🗑️ Borrar caché"):
                            try:
                                os.remove(cache_path)
                                st.warning("Caché eliminado. Re-entrena el modelo.")
                                st.rerun()
                            except:
                                pass
                else:
                    use_cached = False
                    if cache_metadata and cache_metadata.get('reason') == 'expired':
                        st.warning(f"⚠️ Modelo en caché expirado ({cache_metadata['age_days']} días > {max_age_days} días). Se re-entrenará.")
                    else:
                        st.info("📝 No se encontró modelo en caché. Se entrenará desde cero.")
                
                # Ejecutar con o sin caché según elección del usuario
                try:
                    predictor, data_clean, from_cache, metadata = train_or_load_model(
                        ticker=ticker,
                        model_type=model_type,
                        test_size=test_size,
                        use_cache=use_cached,
                        max_age_days=max_age_days,
                        force_retrain=not use_cached
                    )
                    
                    # Guardar en session_state
                    st.session_state['predictor'] = predictor
                    st.session_state['data_clean'] = data_clean
                    
                    # Generar predicciones automáticamente
                    with st.spinner('🎯 Generando recomendaciones de inversión...'):
                        predictions_df = predictor.predict_investment_horizons(
                            data_clean, 
                            show_details=False
                        )
                        st.session_state['predictions_df'] = predictions_df
                        
                        if from_cache:
                            st.success("✅ Recomendaciones generadas usando modelo en caché")
                        else:
                            st.success("✅ Recomendaciones generadas con nuevo modelo entrenado")
                    
                    # Forzar rerun para mostrar las predicciones
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    
            else:
                # ==================== MODO SIN CACHÉ (LEGACY) ====================
                st.warning("⚠️ Sistema de caché no disponible. Entrenando modo tradicional...")
                
                with st.spinner('Descargando histórico completo y preparando datos...'):
                    # Cargar datos históricos completos para ML
                    st.info("📥 Descargando máximo histórico disponible para entrenamiento...")
                    data_full = load_data_for_ml(ticker)
                    
                    if data_full is None or len(data_full) < 50:
                        st.error(f"❌ No se pudieron cargar suficientes datos históricos. Solo se obtuvieron {len(data_full) if data_full is not None else 0} registros.")
                        st.info("💡 Intenta con otro ticker o verifica tu conexión a internet.")
                        return
                    
                    st.success(f"✅ Descargados {len(data_full):,} registros históricos")
                    
                    # Preparar datos
                    engineer = FeatureEngineer()
                    data_featured = engineer.add_technical_indicators(data_full)
                    data_clean = data_featured.dropna()
                    
                    st.info(f"📊 Datos listos: {len(data_clean):,} muestras utilizables tras limpieza")
                    
                    # Crear predictor mejorado
                    predictor = ImprovedStockPredictor(model_type=model_type)
                    
                    # Entrenar todos los horizontes
                    try:
                        trained_count = predictor.train_all_horizons(
                            data_clean, 
                            test_size=test_size
                        )
                        
                        st.success(f"✅ {trained_count} modelos entrenados exitosamente")
                        
                        # Guardar en session_state
                        st.session_state['predictor'] = predictor
                        st.session_state['data_clean'] = data_clean
                        
                        # Generar predicciones automáticamente
                        with st.spinner('🎯 Generando recomendaciones de inversión...'):
                            predictions_df = predictor.predict_investment_horizons(
                                data_clean, 
                                show_details=False
                            )
                            st.session_state['predictions_df'] = predictions_df
                            st.success("✅ Recomendaciones generadas automáticamente")
                        
                        # Forzar rerun para mostrar las predicciones
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error durante el entrenamiento: {str(e)}")
                        st.info("💡 Revisa la consola/terminal para más detalles del error.")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Mostrar predicciones si el modelo está entrenado
        if 'predictor' in st.session_state and st.session_state['predictor'].is_trained:
            predictor = st.session_state['predictor']
            data_clean = st.session_state['data_clean']
            
            st.divider()
            
            # Botón para actualizar predicciones (opcional)
            if st.button('🔄 Actualizar Recomendaciones', type='secondary'):
                with st.spinner('Actualizando predicciones...'):
                    # Hacer predicciones
                    predictions_df = predictor.predict_investment_horizons(
                        data_clean, 
                        show_details=False
                    )
                    
                    st.session_state['predictions_df'] = predictions_df
                    st.success("✅ Recomendaciones actualizadas")
            
            # Mostrar predicciones si existen
            if 'predictions_df' in st.session_state:
                predictions_df = st.session_state['predictions_df']
                
                # Verificar que el DataFrame tiene las columnas esperadas
                required_columns = ['Precio Actual', 'Precio Predicho', 'Cambio %', 'Cambio $']
                if not all(col in predictions_df.columns for col in required_columns):
                    st.error("❌ Error: El DataFrame de predicciones no tiene el formato esperado.")
                    st.info("💡 Intenta entrenar los modelos nuevamente.")
                    if st.checkbox("🔍 Mostrar columnas disponibles"):
                        st.write(predictions_df.columns.tolist())
                else:
                    # ====== SECCIÓN DE ANÁLISIS DE MERCADO ======
                    st.header("💹 Análisis de Valor de Mercado")
                    
                    # Precio actual y estadísticas clave
                    current_price = predictions_df['Precio Actual'].iloc[0]
                    
                    # Mejor y peor escenario
                    best_idx = predictions_df['Cambio %'].idxmax()
                    worst_idx = predictions_df['Cambio %'].idxmin()
                    best_case = predictions_df.loc[best_idx]
                    worst_case = predictions_df.loc[worst_idx]
                
                    # Métricas principales en 4 columnas
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "💰 Precio Actual",
                            f"${current_price:.2f}"
                        )
                    
                    with col2:
                        avg_predicted = predictions_df['Precio Predicho'].mean()
                        avg_change = ((avg_predicted - current_price) / current_price) * 100
                        st.metric(
                            "📊 Precio Promedio Predicho",
                            f"${avg_predicted:.2f}",
                            f"{avg_change:+.2f}%"
                        )
                    
                    with col3:
                        st.metric(
                            f"🎯 Mejor Escenario ({best_case['Horizonte']})",
                            f"${best_case['Precio Predicho']:.2f}",
                            f"{best_case['Cambio %']:+.2f}%",
                            delta_color="normal"
                        )
                    
                    with col4:
                        st.metric(
                            f"⚠️ Peor Escenario ({worst_case['Horizonte']})",
                            f"${worst_case['Precio Predicho']:.2f}",
                            f"{worst_case['Cambio %']:+.2f}%",
                            delta_color="inverse"
                        )
                    
                    # Proyección de ganancias/pérdidas por inversión
                    st.divider()
                    st.subheader("💵 Proyección de Inversión")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        investment_amount = st.number_input(
                            "Cantidad a invertir ($)",
                            min_value=100.0,
                            max_value=1000000.0,
                            value=10000.0,
                            step=100.0
                        )
                    
                    with col2:
                        # Calcular cuántas acciones se pueden comprar
                        shares = investment_amount / current_price
                        
                        st.info(f"**Con ${investment_amount:,.2f} puedes comprar:** {shares:.2f} acciones")
                        
                        # Cálculo para diferentes horizontes
                        best_return = shares * best_case['Precio Predicho']
                        worst_return = shares * worst_case['Precio Predicho']
                        avg_return = shares * avg_predicted
                        
                        col2_1, col2_2, col2_3 = st.columns(3)
                        
                        with col2_1:
                            st.metric(
                                f"Mejor caso ({best_case['Horizonte']})",
                                f"${best_return:,.2f}",
                                f"${best_return - investment_amount:+,.2f}"
                            )
                        
                        with col2_2:
                            st.metric(
                                "Promedio todos horizontes",
                                f"${avg_return:,.2f}",
                                f"${avg_return - investment_amount:+,.2f}"
                            )
                        
                        with col2_3:
                            st.metric(
                                f"Peor caso ({worst_case['Horizonte']})",
                                f"${worst_return:,.2f}",
                                f"${worst_return - investment_amount:+,.2f}"
                            )
                    
                    # Tabla de proyección por horizonte
                    st.divider()
                    st.subheader("📈 Proyección Detallada por Horizonte")
                    
                    # Crear tabla de proyección
                    projection_df = predictions_df[['Horizonte', 'Días', 'Señal', 'Precio Predicho', 'Cambio %']].copy()
                    projection_df['Valor Inversión'] = shares * projection_df['Precio Predicho']
                    projection_df['Ganancia/Pérdida'] = projection_df['Valor Inversión'] - investment_amount
                    projection_df['ROI %'] = ((projection_df['Valor Inversión'] - investment_amount) / investment_amount) * 100
                    
                    # Formatear para display
                    display_projection = projection_df.copy()
                    display_projection['Precio Predicho'] = display_projection['Precio Predicho'].apply(lambda x: f"${x:.2f}")
                    display_projection['Cambio %'] = display_projection['Cambio %'].apply(lambda x: f"{x:+.2f}%")
                    display_projection['Valor Inversión'] = display_projection['Valor Inversión'].apply(lambda x: f"${x:,.2f}")
                    display_projection['Ganancia/Pérdida'] = display_projection['Ganancia/Pérdida'].apply(lambda x: f"${x:+,.2f}")
                    display_projection['ROI %'] = display_projection['ROI %'].apply(lambda x: f"{x:+.2f}%")
                    
                    st.dataframe(
                        display_projection,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Horizonte": st.column_config.TextColumn("Horizonte", width="medium"),
                            "Días": st.column_config.NumberColumn("Días", width="small"),
                            "Señal": st.column_config.TextColumn("Señal", width="small"),
                            "Precio Predicho": st.column_config.TextColumn("Precio", width="small"),
                            "Cambio %": st.column_config.TextColumn("Cambio", width="small"),
                            "Valor Inversión": st.column_config.TextColumn("Valor Final", width="medium"),
                            "Ganancia/Pérdida": st.column_config.TextColumn("Ganancia/Pérdida", width="medium"),
                            "ROI %": st.column_config.TextColumn("ROI", width="small"),
                        }
                    )
                    
                    st.divider()
                    st.subheader("📊 Recomendaciones Detalladas por Horizonte")
                    
                    # Formatear DataFrame para mostrar
                    display_df = predictions_df.copy()
                    display_df['Precio Actual'] = display_df['Precio Actual'].apply(lambda x: f"${x:.2f}")
                    display_df['Precio Predicho'] = display_df['Precio Predicho'].apply(lambda x: f"${x:.2f}")
                    display_df['Cambio $'] = display_df['Cambio $'].apply(lambda x: f"${x:+.2f}")
                    display_df['Cambio %'] = display_df['Cambio %'].apply(lambda x: f"{x:+.2f}%")
                    display_df['Prob. Subida'] = display_df['Prob. Subida'].apply(lambda x: f"{x:.1f}%")
                    display_df['Prob. Bajada'] = display_df['Prob. Bajada'].apply(lambda x: f"{x:.1f}%")
                    display_df['Confianza'] = display_df['Confianza'].apply(lambda x: f"{x:.1f}%")
                    display_df['Test Acc'] = display_df['Test Acc'].apply(lambda x: f"{x:.1f}%")
                    
                    # Mostrar tabla completa formateada
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Resumen visual
                    st.divider()
                    st.subheader("📈 Resumen de Señales")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    comprar = len(predictions_df[predictions_df['Señal'].str.contains('COMPRAR')])
                    vender = len(predictions_df[predictions_df['Señal'].str.contains('VENDER')])
                    total = len(predictions_df)
                    
                    with col1:
                        st.metric(
                            "Señales COMPRAR",
                            f"{comprar}/{total}",
                            f"{(comprar/total)*100:.0f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Señales VENDER",
                            f"{vender}/{total}",
                            f"{(vender/total)*100:.0f}%"
                        )
                    
                    with col3:
                        if comprar > vender:
                            sentiment = "ALCISTA 📈"
                            delta_color = "normal"
                        elif vender > comprar:
                            sentiment = "BAJISTA 📉"
                            delta_color = "inverse"
                        else:
                            sentiment = "NEUTRAL ➡️"
                            delta_color = "off"
                        
                        st.metric(
                            "Sentimiento General",
                            sentiment
                        )
                    
                    # Gráfico de evolución de precio predicho
                    st.divider()
                    st.subheader("💰 Evolución de Precio Predicho por Horizonte")
                    
                    fig_price = go.Figure()
                    
                    # Línea base (precio actual)
                    fig_price.add_hline(
                        y=current_price,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text=f"Precio Actual: ${current_price:.2f}",
                        annotation_position="right"
                    )
                    
                    # Precios predichos
                    fig_price.add_trace(go.Scatter(
                        x=predictions_df['Horizonte'],
                        y=predictions_df['Precio Predicho'],
                        mode='lines+markers',
                        name='Precio Predicho',
                        line=dict(width=3, color='blue'),
                        marker=dict(
                            size=12,
                            color=predictions_df['Cambio %'],
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title="Cambio %", x=1.15)
                        ),
                        text=predictions_df.apply(
                            lambda row: f"{row['Horizonte']}<br>Precio: ${row['Precio Predicho']:.2f}<br>Cambio: {row['Cambio %']:+.2f}%",
                            axis=1
                        ),
                        hovertemplate='%{text}<extra></extra>'
                    ))
                    
                    fig_price.update_layout(
                        title='Precio Predicho por Horizonte Temporal',
                        xaxis_title='Horizonte',
                        yaxis_title='Precio ($)',
                        height=500,
                        hovermode='x unified',
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_price, use_container_width=True)
                    
                    # Gráfico de ROI esperado
                    st.subheader("📊 ROI Esperado por Horizonte")
                    
                    # Calcular ROI para el investment_amount actual
                    roi_data = predictions_df.copy()
                    roi_data['ROI %'] = ((roi_data['Precio Predicho'] - current_price) / current_price) * 100
                    
                    fig_roi = go.Figure()
                    
                    # Línea de cero
                    fig_roi.add_hline(
                        y=0,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Break-even",
                        annotation_position="left"
                    )
                    
                    # Barras de ROI
                    colors = roi_data['ROI %'].apply(lambda x: 'green' if x > 0 else 'red')
                    
                    fig_roi.add_trace(go.Bar(
                        x=roi_data['Horizonte'],
                        y=roi_data['ROI %'],
                        marker_color=colors,
                        text=roi_data['ROI %'].apply(lambda x: f"{x:+.1f}%"),
                        textposition='outside',
                        name='ROI',
                        hovertemplate='%{x}<br>ROI: %{y:.2f}%<extra></extra>'
                    ))
                    
                    fig_roi.update_layout(
                        title='Retorno de Inversión (ROI) Esperado',
                        xaxis_title='Horizonte Temporal',
                        yaxis_title='ROI (%)',
                        height=450,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_roi, use_container_width=True)
                    
                    # Gráfico de barras por horizonte
                    st.divider()
                    st.subheader("📊 Confianza por Horizonte")
                    
                    # Preparar datos para los gráficos
                    plot_df = predictions_df.copy()
                    plot_df['Color'] = plot_df['Señal'].apply(
                        lambda x: 'green' if 'COMPRAR' in x else 'red'
                    )
                    
                    # Gráfico de confianza
                    fig_conf = go.Figure()
                    fig_conf.add_trace(go.Bar(
                        x=plot_df['Horizonte'],
                        y=plot_df['Confianza'],
                        marker_color=plot_df['Color'],
                        text=plot_df['Confianza'].apply(lambda x: f"{x:.1f}%"),
                        textposition='outside',
                        name='Confianza'
                    ))
                    fig_conf.update_layout(
                        title='Nivel de Confianza por Horizonte',
                        xaxis_title='Horizonte Temporal',
                        yaxis_title='Confianza (%)',
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)
                    
                    # Gráfico de probabilidades apiladas
                    st.subheader("📊 Probabilidades de Movimiento")
                    
                    fig_prob = go.Figure()
                    fig_prob.add_trace(go.Bar(
                        x=plot_df['Horizonte'],
                        y=plot_df['Prob. Subida'],
                        name='Prob. Subida',
                        marker_color='green',
                        text=plot_df['Prob. Subida'].apply(lambda x: f"{x:.1f}%"),
                        textposition='inside'
                    ))
                    fig_prob.add_trace(go.Bar(
                        x=plot_df['Horizonte'],
                        y=plot_df['Prob. Bajada'],
                        name='Prob. Bajada',
                        marker_color='red',
                        text=plot_df['Prob. Bajada'].apply(lambda x: f"{x:.1f}%"),
                        textposition='inside'
                    ))
                    fig_prob.update_layout(
                        barmode='stack',
                        title='Distribución de Probabilidades',
                        xaxis_title='Horizonte Temporal',
                        yaxis_title='Probabilidad (%)',
                        height=400
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                    # Análisis detallado por categoría de horizonte
                    st.divider()
                    st.subheader("🔍 Análisis Detallado")
                    
                    tab1, tab2, tab3 = st.tabs(["Corto Plazo", "Medio Plazo", "Largo Plazo"])
                    
                    with tab1:
                        corto_mask = predictions_df['Días'] <= 63
                        if corto_mask.any():
                            st.dataframe(display_df[corto_mask], use_container_width=True, hide_index=True)
                            avg_change = predictions_df[corto_mask]['Cambio %'].mean()
                            st.info(f"💡 **Corto plazo** (hasta 3 meses): Ideal para trading activo. Cambio promedio esperado: {avg_change:+.2f}%")
                        else:
                            st.warning("No hay datos de corto plazo")
                    
                    with tab2:
                        medio_mask = (predictions_df['Días'] > 63) & (predictions_df['Días'] <= 756)
                        if medio_mask.any():
                            st.dataframe(display_df[medio_mask], use_container_width=True, hide_index=True)
                            avg_change = predictions_df[medio_mask]['Cambio %'].mean()
                            st.info(f"💡 **Medio plazo** (3 meses - 3 años): Balance entre riesgo y retorno. Cambio promedio esperado: {avg_change:+.2f}%")
                        else:
                            st.warning("No hay datos de medio plazo")
                    
                    with tab3:
                        largo_mask = predictions_df['Días'] > 756
                        if largo_mask.any():
                            st.dataframe(display_df[largo_mask], use_container_width=True, hide_index=True)
                            avg_change = predictions_df[largo_mask]['Cambio %'].mean()
                            st.info(f"💡 **Largo plazo** (más de 3 años): Inversión estratégica a largo plazo. Cambio promedio esperado: {avg_change:+.2f}%")
                        else:
                            st.warning("No hay datos de largo plazo")
                    
                    # Botón para ver importancia de features
                    st.divider()
                    if st.checkbox("🔍 Mostrar Importancia de Features"):
                        horizon_options = {
                            f"{row['Horizonte']} ({row['Días']} días)": row['Días'] 
                            for _, row in predictions_df.iterrows()
                        }
                        
                        selected_horizon_name = st.selectbox(
                            "Selecciona un horizonte:",
                            options=list(horizon_options.keys())
                        )
                        
                        selected_horizon_days = horizon_options[selected_horizon_name]
                        
                        if selected_horizon_days in predictor.models:
                            model_data = predictor.models[selected_horizon_days]
                            model = model_data['model']
                            
                            if hasattr(model, 'feature_importances_'):
                                importance_df = pd.DataFrame({
                                    'Feature': predictor.feature_cols,
                                    'Importancia': model.feature_importances_
                                }).sort_values('Importancia', ascending=False).head(15)
                                
                                fig_imp = px.bar(
                                    importance_df,
                                    x='Importancia',
                                    y='Feature',
                                    orientation='h',
                                    title=f'Top 15 Features - {selected_horizon_name}'
                                )
                                fig_imp.update_layout(height=500)
                                st.plotly_chart(fig_imp, use_container_width=True)
                            else:
                                st.warning("Este modelo no soporta feature importance")
                    
                    # Guardar modelos y gestión de caché
                    st.divider()
                    
                    if CACHE_AVAILABLE:
                        st.subheader("💾 Gestión de Modelos")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("💾 Guardar en Backup Manual"):
                                try:
                                    predictor.save_models('models/saved/improved_predictor.pkl')
                                    st.success("✅ Backup guardado en models/saved/")
                                except Exception as e:
                                    st.error(f"Error guardando backup: {e}")
                        
                        with col2:
                            # Listar todos los modelos en caché
                            cache_dir = 'models/cache'
                            if os.path.exists(cache_dir):
                                cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
                                
                                if cache_files:
                                    with st.expander(f"📂 Ver Modelos en Caché ({len(cache_files)})"):
                                        for filename in sorted(cache_files):
                                            filepath = os.path.join(cache_dir, filename)
                                            mtime = os.path.getmtime(filepath)
                                            size_mb = os.path.getsize(filepath) / (1024 * 1024)
                                            from datetime import datetime
                                            date = datetime.fromtimestamp(mtime)
                                            age = (datetime.now() - date).days
                                            
                                            st.text(f"📄 {filename}")
                                            st.text(f"   📅 {date.strftime('%Y-%m-%d %H:%M')} ({age}d)")
                                            st.text(f"   💾 {size_mb:.2f} MB")
                                            st.text("")
                                else:
                                    st.info("No hay modelos en caché")
                    else:
                        if st.button("💾 Guardar Modelos Entrenados"):
                            try:
                                predictor.save_models('models/saved/improved_predictor.pkl')
                                st.success("✅ Modelos guardados exitosamente")
                            except Exception as e:
                                st.error(f"Error guardando modelos: {e}")
        
    except Exception as e:
        st.error(f"❌ Error en predicciones: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.info("Asegúrate de tener scikit-learn instalado: pip install scikit-learn")

if __name__ == "__main__":
    main()