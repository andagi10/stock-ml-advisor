"""
Módulo para crear features técnicas para ML
"""
import pandas as pd
import numpy as np
from typing import List


class FeatureEngineer:
    """
    Crea indicadores técnicos y features para los modelos
    """
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade indicadores técnicos al DataFrame
        
        Args:
            df: DataFrame con datos OHLCV
        
        Returns:
            DataFrame con indicadores añadidos
        """
        df = df.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # RSI (Relative Strength Index)
        df['rsi'] = FeatureEngineer.calculate_rsi(df['close'])
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price position
        df['price_position'] = (df['close'] - df['close'].rolling(window=20).min()) / \
                                (df['close'].rolling(window=20).max() - df['close'].rolling(window=20).min())
        
        return df
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcula el RSI (Relative Strength Index)
        
        Args:
            prices: Serie de precios
            period: Período para el cálculo
        
        Returns:
            Serie con valores RSI
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def create_target_variable(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """
        Crea variable objetivo (target) para predicción
        
        Args:
            df: DataFrame con datos
            horizon: Días hacia adelante para predecir
        
        Returns:
            DataFrame con target añadido
        """
        df = df.copy()
        
        # Retorno futuro
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        
        # Clasificación: 1 si sube, 0 si baja
        df['target'] = (df['future_return'] > 0).astype(int)
        
        # Clasificación multi-clase
        df['target_multiclass'] = pd.cut(
            df['future_return'],
            bins=[-np.inf, -0.02, 0.02, np.inf],
            labels=['sell', 'hold', 'buy']
        )
        
        return df
    
    @staticmethod
    def prepare_features(df: pd.DataFrame, feature_cols: List[str] = None) -> tuple:
        """
        Prepara features y target para entrenamiento
        
        Args:
            df: DataFrame con todos los datos
            feature_cols: Lista de columnas a usar como features
        
        Returns:
            Tuple (X, y) con features y target
        """
        if feature_cols is None:
            # Seleccionar automáticamente features numéricas
            feature_cols = [
                'returns', 'sma_20', 'sma_50', 'rsi', 
                'macd', 'macd_signal', 'bb_width', 
                'volatility', 'volume_ratio', 'price_position'
            ]
        
        # Eliminar NaN
        df_clean = df.dropna()
        
        X = df_clean[feature_cols]
        y = df_clean['target']
        
        return X, y


def main():
    """
    Función de testing
    """
    print("🔧 Testing Feature Engineering...")
    
    # Crear datos de ejemplo
    from data.collectors.market_data import MarketDataCollector
    
    collector = MarketDataCollector()
    data = collector.download_ticker('^GSPC', period='1y')
    
    if data is not None:
        # Crear features
        engineer = FeatureEngineer()
        data_with_features = engineer.add_technical_indicators(data)
        data_with_target = engineer.create_target_variable(data_with_features)
        
        print(f"\n✅ Features creadas: {len(data_with_features.columns)} columnas")
        print(f"📊 Primeras features:")
        print(data_with_features[['close', 'sma_20', 'rsi', 'macd']].tail())
        
        # Preparar para ML
        X, y = engineer.prepare_features(data_with_target)
        print(f"\n🎯 Dataset para ML:")
        print(f"   Features (X): {X.shape}")
        print(f"   Target (y): {y.shape}")
        print(f"   Distribución target: {y.value_counts().to_dict()}")


if __name__ == "__main__":
    main()