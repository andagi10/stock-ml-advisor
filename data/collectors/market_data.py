"""
Módulo para recolectar datos de mercado desde Yahoo Finance
"""
import yfinance as yf
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import os


class MarketDataCollector:
    """
    Recolecta datos históricos de múltiples mercados
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Inicializa el recolector con la configuración
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.markets = self.config['markets']
        self.data_config = self.config['data']
        
        # Crear directorios si no existen
        os.makedirs(self.data_config['raw_data_path'], exist_ok=True)
        os.makedirs(self.data_config['processed_data_path'], exist_ok=True)
    
    def download_ticker(self, ticker: str, period: str = None) -> Optional[pd.DataFrame]:
        """
        Descarga datos de un ticker específico
        
        Args:
            ticker: Símbolo del ticker (ej: '^GSPC', 'SAN.MC')
            period: Período de datos ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        
        Returns:
            DataFrame con los datos o None si falla
        """
        if period is None:
            period = self.data_config['lookback_period']
        
        try:
            print(f"📥 Descargando {ticker}...")
            data = yf.download(ticker, period=period, progress=False)
            
            if data.empty:
                print(f"⚠️  No hay datos para {ticker}")
                return None
            
            # Limpiar nombres de columnas
            data.columns = data.columns.str.lower()
            
            # Añadir ticker como columna
            data['ticker'] = ticker
            
            print(f"✅ {ticker}: {len(data)} días descargados")
            return data
            
        except Exception as e:
            print(f"❌ Error descargando {ticker}: {e}")
            return None
    
    def download_all_markets(self) -> Dict[str, pd.DataFrame]:
        """
        Descarga datos de todos los mercados configurados
        
        Returns:
            Diccionario con los datos de cada mercado
        """
        all_data = {}
        
        for market_name, tickers in self.markets.items():
            print(f"\n🌍 Mercado: {market_name.upper()}")
            market_data = {}
            
            for ticker_info in tickers:
                ticker = ticker_info['ticker']
                name = ticker_info['name']
                
                data = self.download_ticker(ticker)
                
                if data is not None:
                    market_data[name] = data
                    
                    # Guardar datos crudos si está configurado
                    if self.data_config.get('save_raw_data', True):
                        filename = f"{self.data_config['raw_data_path']}/{ticker.replace('^', '').replace('.', '_')}.csv"
                        data.to_csv(filename)
                        print(f"💾 Guardado en: {filename}")
            
            all_data[market_name] = market_data
        
        return all_data
    
    def get_latest_price(self, ticker: str) -> Optional[float]:
        """
        Obtiene el precio más reciente de un ticker
        
        Args:
            ticker: Símbolo del ticker
        
        Returns:
            Precio actual o None si falla
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('currentPrice') or info.get('regularMarketPrice')
        except Exception as e:
            print(f"❌ Error obteniendo precio de {ticker}: {e}")
            return None
    
    def get_market_info(self, ticker: str) -> Dict:
        """
        Obtiene información fundamental de un ticker
        
        Args:
            ticker: Símbolo del ticker
        
        Returns:
            Diccionario con información del mercado
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                '52w_high': info.get('fiftyTwoWeekHigh', 0),
                '52w_low': info.get('fiftyTwoWeekLow', 0),
            }
        except Exception as e:
            print(f"❌ Error obteniendo info de {ticker}: {e}")
            return {}


def main():
    """
    Función principal para testing
    """
    print("=" * 60)
    print("📊 STOCK ML ADVISOR - Data Collector")
    print("=" * 60)
    
    # Crear instancia del colector
    collector = MarketDataCollector()
    
    # Descargar todos los datos
    data = collector.download_all_markets()
    
    # Resumen
    print("\n" + "=" * 60)
    print("📈 RESUMEN")
    print("=" * 60)
    
    total_tickers = 0
    for market_name, market_data in data.items():
        print(f"\n{market_name.upper()}:")
        for name, df in market_data.items():
            print(f"  • {name}: {len(df)} días")
            total_tickers += 1
    
    print(f"\n✅ Total: {total_tickers} tickers descargados")
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()