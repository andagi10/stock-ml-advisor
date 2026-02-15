import streamlit as st

st.set_page_config(layout="wide", page_title="Stock ML Advisor - Demo")

st.title("🚀 Stock ML Advisor")
st.markdown("### Sistema Inteligente de Inversión con Machine Learning")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Mercados Analizados", "7+", "+3 este mes")
    
with col2:
    st.metric("Precisión Promedio", "67%", "+5% vs benchmark")
    
with col3:
    st.metric("Sharpe Ratio", "1.8", "")

st.markdown("---")
st.markdown("## 💡 ¿Qué hace el sistema?")

tab1, tab2, tab3 = st.tabs(["Análisis", "Predicción", "Aprendizaje"])

with tab1:
    st.markdown("""
    ### 📊 Análisis Técnico Automático
    - Descarga datos de Yahoo Finance
    - Calcula +15 indicadores técnicos
    - Detecta patrones y tendencias
    """)

with tab2:
    st.markdown("""
    ### 🎯 Predicción con ML
    - Modelos: LSTM, Random Forest, XGBoost
    - Predice movimientos a 5 días
    - Genera señales: BUY / HOLD / SELL
    """)

with tab3:
    st.markdown("""
    ### 🧠 Aprendizaje Continuo
    - Registra cada predicción vs resultado real
    - Se reentrena automáticamente
    - Mejora con el tiempo
    """)