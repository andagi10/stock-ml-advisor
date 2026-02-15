"""
Dashboard interactivo para Stock ML Advisor
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

# Añadir raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from data.collectors.market_data import MarketDataCollector
from data.processors.feature_engineering import FeatureEngineer


# Configuración de la página
st.set_page_config(
    page_title="Stock ML Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_data(ticker, period):
    """Carga y procesa datos"""
    collector = MarketDataCollector()
    data = collector.download_ticker(ticker, period=period)
    
    if data is not None:
        engineer = FeatureEngineer()
        data = engineer.add_technical_indicators(data)
        data = engineer.create_target_variable(data, horizon=5)
    
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
    
    selected_market = st.sidebar.selectbox(
        'Selecciona un mercado:',
        options=list(market_options.keys())
    )
    
    ticker = market_options[selected_market]
    
    # Período
    period = st.sidebar.selectbox(
        'Período de análisis:',
        options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=3  # Default: 1y
    )
    
    # Botón de carga
    if st.sidebar.button('🔄 Cargar Datos', type='primary'):
        with st.spinner(f'Descargando datos de {selected_market}...'):
            data = load_data(ticker, period)
            st.session_state['data'] = data
            st.session_state['ticker'] = ticker
            st.session_state['market_name'] = selected_market
    
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
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Gráfico Principal",
            "📉 Indicadores Técnicos",
            "📈 Análisis Estadístico",
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


if __name__ == "__main__":
    main()