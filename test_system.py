"""
Script simple para probar todo el sistema
"""
from data.collectors.market_data import MarketDataCollector
from data.processors.feature_engineering import FeatureEngineer
import pandas as pd


def test_data_collection():
    """Prueba la recolección de datos"""
    print("\n" + "="*60)
    print("TEST 1: Recolección de Datos")
    print("="*60)
    
    collector = MarketDataCollector()
    
    # Probar con S&P500
    data = collector.download_ticker('^GSPC', period='1y')
    
    if data is not None:
        print(f"✅ Datos descargados: {len(data)} días")
        print(f"📊 Columnas: {list(data.columns)}")
        print(f"\n📈 Últimos 5 días:")
        print(data[['close', 'volume']].tail())
        return data
    else:
        print("❌ Error en descarga")
        return None


def test_feature_engineering(data):
    """Prueba la creación de features"""
    print("\n" + "="*60)
    print("TEST 2: Feature Engineering")
    print("="*60)
    
    engineer = FeatureEngineer()
    
    # Añadir indicadores
    data_with_features = engineer.add_technical_indicators(data)
    print(f"✅ Features técnicas añadidas")
    
    # Crear target
    data_with_target = engineer.create_target_variable(data_with_features, horizon=5)
    print(f"✅ Variable target creada")
    
    # Preparar para ML
    X, y = engineer.prepare_features(data_with_target)
    print(f"\n📊 Dataset preparado:")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Target distribución: {y.value_counts().to_dict()}")
    
    return X, y


def main():
    """Ejecutar todas las pruebas"""
    print("\n🚀 INICIANDO PRUEBAS DEL SISTEMA")
    print("="*60)
    
    # Test 1: Datos
    data = test_data_collection()
    
    if data is not None:
        # Test 2: Features
        X, y = test_feature_engineering(data)
        
        print("\n" + "="*60)
        print("✅ TODAS LAS PRUEBAS COMPLETADAS")
        print("="*60)
        print("\n🎯 Próximos pasos:")
        print("   1. Entrenar un modelo simple")
        print("   2. Hacer backtesting")
        print("   3. Implementar sistema de tracking")
    else:
        print("\n❌ Las pruebas fallaron")


if __name__ == "__main__":
    main()