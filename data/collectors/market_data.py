"""
Módulo para recolectar datos de mercado desde Yahoo Finance
"""
import yfinance as yf
import pandas as pd
import yaml
import requests
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import os


class MarketDataCollector:
    """
    Recolecta datos históricos de múltiples mercados
    """
    
    def __init__(self, config_path: str = "config/settings.yaml", max_retries: int = 3, delay: int = 2):
        """
        Inicializa el recolector con la configuración
        
        Args:
            config_path: Ruta al archivo de configuración
            max_retries: Número máximo de reintentos por descarga
            delay: Segundos entre reintentos
        """
        self.max_retries = max_retries
        self.delay = delay
        
        # Configurar sesión HTTP con headers robustos
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        
        # Cargar configuración
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.markets = self.config['markets']
        self.data_config = self.config['data']
        
        # Crear directorios si no existen
        os.makedirs(self.data_config['raw_data_path'], exist_ok=True)
        os.makedirs(self.data_config['processed_data_path'], exist_ok=True)
    
    def download_ticker(self, ticker: str, period: str = None) -> Optional[pd.DataFrame]:
        """
        Descarga datos de un ticker específico con múltiples estrategias de respaldo
        
        Args:
            ticker: Símbolo del ticker (ej: '^GSPC', 'SAN.MC')
            period: Período de datos ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        
        Returns:
            DataFrame con los datos o None si falla
        """
        if period is None:
            period = self.data_config['lookback_period']
        
        print(f"📥 Descargando {ticker}...")
        
        # ESTRATEGIA 1: yf.download con sesión y reintentos
        for attempt in range(self.max_retries):
            try:
                data = yf.download(
                    ticker, 
                    period=period, 
                    progress=False,
                    session=self.session,
                    ignore_tz=True
                )
                
                if data is not None and not data.empty:
                    data = self._clean_and_standardize(data, ticker)
                    print(f"✅ {ticker}: {len(data)} días descargados")
                    return data
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"⚠️  Intento {attempt + 1}/{self.max_retries} falló, reintentando en {self.delay}s...")
                    time.sleep(self.delay)
                else:
                    print(f"❌ yf.download falló después de {self.max_retries} intentos: {e}")
        
        # ESTRATEGIA 2: Usar objeto Ticker como respaldo
        try:
            print(f"🔄 Intentando método alternativo para {ticker}...")
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period=period, auto_adjust=True)
            
            if data is not None and not data.empty:
                data = self._clean_and_standardize(data, ticker)
                print(f"✅ {ticker}: {len(data)} días descargados (método alternativo)")
                return data
                
        except Exception as e:
            print(f"❌ Método alternativo también falló: {e}")
        
        # ESTRATEGIA 3: Intentar con período más corto como último recurso
        if period in ['5y', '10y', 'max']:
            try:
                print(f"🔄 Último intento con período reducido (1y) para {ticker}...")
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(period='1y', auto_adjust=True)
                
                if data is not None and not data.empty:
                    data = self._clean_and_standardize(data, ticker)
                    print(f"✅ {ticker}: {len(data)} días descargados (período reducido)")
                    return data
            except Exception as e:
                print(f"❌ Último intento falló: {e}")
        
        print(f"⚠️  No hay datos disponibles para {ticker}")
        return None
    
    def _clean_and_standardize(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Limpia y estandariza el formato de los datos
        
        Args:
            data: DataFrame crudo de yfinance
            ticker: Símbolo del ticker
            
        Returns:
            DataFrame limpio y estandarizado
        """
        # Copiar para no modificar el original
        df = data.copy()
        
        # Limpiar nombres de columnas a minúsculas
        df.columns = df.columns.str.lower()
        
        # Renombrar columnas con espacios o caracteres especiales
        column_mapping = {
            'adj close': 'adj_close',
            'adjusted close': 'adj_close'
        }
        df = df.rename(columns=column_mapping)
        
        # Añadir ticker como columna
        df['ticker'] = ticker
        
        # Asegurar que el índice es datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Ordenar por fecha
        df = df.sort_index()
        
        # Eliminar duplicados
        df = df[~df.index.duplicated(keep='last')]
        
        # Eliminar filas con todos NaN (excepto ticker)
        df = df.dropna(how='all', subset=[col for col in df.columns if col != 'ticker'])
        
        return df
    
    def download_all_markets(self) -> Dict[str, pd.DataFrame]:
        """
        Descarga datos de todos los mercados configurados
        
        Returns:
            Diccionario con los datos de cada mercado
        """
        all_data = {}
        
        for market_name, tickers in self.markets.items():
            print(f"\n🌍 Mercado: {market_name.upper()}")
            print("-" * 60)
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
                
                # Pequeña pausa entre descargas para evitar rate limiting
                time.sleep(0.5)
            
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
            # Intentar obtener precio de info
            stock = yf.Ticker(ticker)
            info = stock.info
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if price:
                return float(price)
            
            # Si no hay precio en info, usar history
            data = self.download_ticker(ticker, period='5d')
            if data is not None and not data.empty:
                return float(data['close'].iloc[-1])
                
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
    
    try:
        # Crear instancia del colector
        collector = MarketDataCollector()
        
        # Descargar todos los datos
        data = collector.download_all_markets()
        
        # Resumen
        print("\n" + "=" * 60)
        print("📈 RESUMEN")
        print("=" * 60)
        
        total_tickers = 0
        total_days = 0
        
        for market_name, market_data in data.items():
            print(f"\n{market_name.upper()}:")
            for name, df in market_data.items():
                days = len(df)
                print(f"  • {name}: {days} días")
                total_tickers += 1
                total_days += days
        
        print(f"\n✅ Total: {total_tickers} tickers descargados")
        print(f"📊 Total datos: {total_days:,} días")
        print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except FileNotFoundError:
        print("\n❌ Error: No se encontró el archivo config/settings.yaml")
        print("💡 Asegúrate de tener la estructura correcta de directorios:")
        print("   config/")
        print("   └── settings.yaml")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()