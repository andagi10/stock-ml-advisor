"""
Sistema de predicciones con Machine Learning
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
from datetime import datetime, timedelta


class StockPredictor:
    """
    Sistema de predicción de movimientos de mercado
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_cols = None
        self.is_trained = False
        
    def train(self, X, y, test_size=0.2):
        """
        Entrena el modelo
        
        Args:
            X: Features (DataFrame)
            y: Target (Series)
            test_size: Porcentaje para test
        """
        print("\n🧠 Entrenando modelo...")
        
        # Guardar nombres de features
        self.feature_cols = list(X.columns)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Entrenar modelo
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        
        self.model.fit(X_train, y_train)
        
        # Evaluar
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Modelo entrenado")
        print(f"📊 Accuracy en test: {accuracy:.2%}")
        print(f"📈 Samples de entrenamiento: {len(X_train)}")
        print(f"📉 Samples de test: {len(X_test)}")
        
        # Report detallado
        print("\n📋 Reporte de clasificación:")
        print(classification_report(y_test, y_pred, target_names=['Bajada', 'Subida']))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n🔝 Top 5 features más importantes:")
            print(importance.head())
        
        self.is_trained = True
        return accuracy
    
    def predict(self, X):
        """
        Hace predicciones
        
        Args:
            X: Features (DataFrame o últimas filas)
        
        Returns:
            Predicción: 0 = Bajada, 1 = Subida
        """
        if not self.is_trained:
            raise Exception("❌ El modelo no está entrenado. Llama a train() primero.")
        
        # Asegurar que tiene las features correctas
        X = X[self.feature_cols]
        
        prediction = self.model.predict(X)
        probability = self.model.predict_proba(X)
        
        return prediction, probability
    
    def predict_next_days(self, data_with_features, days=15):
        """
        Predice los próximos N días
        
        Args:
            data_with_features: DataFrame con features calculadas
            days: Número de días a predecir
        
        Returns:
            DataFrame con predicciones
        """
        if not self.is_trained:
            raise Exception("❌ El modelo no está entrenado.")
        
        # Tomar las últimas filas (más recientes)
        recent_data = data_with_features[self.feature_cols].tail(days)
        future_predictions = []

        predictions = []
        probabilities = []
        last_row = data_with_features[self.feature_cols].iloc[-1:].copy()
        
        last_date = pd.Timestamp.today().normalize()
        
        for i in range(days):
            next_date = last_date + pd.Timedelta(days=1)
        
            # Hacer predicción usando la última fila
            pred, prob = self.predict(last_row)
        
            future_predictions.append({
            'date': next_date,
            'prediction': pred[0],
            'prob_down': prob[0][0],
            'prob_up': prob[0][1],
            'signal': '📉 VENDER' if pred[0]==0 else '📈 COMPRAR',
            'confidence': max(prob[0][0], prob[0][1])
            })
        
        # Actualizar last_row si quieres usar predicción para features del siguiente día
        # Por ejemplo, si usas medias móviles, RSI, etc., necesitarías recalcular aquí
        # last_row = update_features(last_row, pred)  <-- si aplicable
        
            last_date = next_date
        return pd.DataFrame(future_predictions)

    def save_model(self, filepath='models/saved/stock_predictor.pkl'):
        """Guarda el modelo entrenado"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_cols': self.feature_cols,
            'model_type': self.model_type,
            'trained_date': datetime.now()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Modelo guardado en: {filepath}")
    
    def load_model(self, filepath='models/saved/stock_predictor.pkl'):
        """Carga un modelo guardado"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_cols = model_data['feature_cols']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        print(f"📂 Modelo cargado desde: {filepath}")
        print(f"📅 Entrenado el: {model_data['trained_date']}")


def main():
    """Test del predictor"""
    print("="*60)
    print("🔮 SISTEMA DE PREDICCIONES")
    print("="*60)
    
    from data.collectors.market_data import MarketDataCollector
    from data.processors.feature_engineering import FeatureEngineer
    
    # 1. Descargar datos
    print("\n1️⃣ Descargando datos...")
    collector = MarketDataCollector()
    data = collector.download_ticker('^GSPC', period='2y')
    
    # 2. Crear features
    print("\n2️⃣ Creando features...")
    engineer = FeatureEngineer()
    data = engineer.add_technical_indicators(data)
    data = engineer.create_target_variable(data, horizon=5)
    
    # 3. Preparar datos
    print("\n3️⃣ Preparando datos...")
    X, y = engineer.prepare_features(data)
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    
    # 4. Entrenar modelo
    print("\n4️⃣ Entrenando modelo...")
    predictor = StockPredictor(model_type='random_forest')
    accuracy = predictor.train(X, y)
    
    # 5. Hacer predicciones
    print("\n5️⃣ Predicciones para los próximos 5 días:")
    predictions = predictor.predict_next_days(data.dropna(), days=15)
    print(predictions.to_string(index=False))
    
    # 6. Guardar modelo
    print("\n6️⃣ Guardando modelo...")
    predictor.save_model()
    
    print("\n" + "="*60)
    print("✅ SISTEMA DE PREDICCIONES COMPLETADO")
    print("="*60)


if __name__ == "__main__":
    main()