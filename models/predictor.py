"""
Sistema Mejorado de Predicciones con Machine Learning
Soporte para horizontes: 1 día -> 20 años
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ImprovedStockPredictor:
    """
    Sistema mejorado de predicción de movimientos de mercado
    con soporte para múltiples horizontes temporales
    """
    
    # Definir horizontes estándar en días de trading
    HORIZONS = {
        '1_dia': 1,
        '1_semana': 5,
        '1_mes': 21,
        '3_meses': 63,
        '6_meses': 126,
        '1_ano': 252,
        '3_anos': 756,
        '5_anos': 1260,
        '10_anos': 2520,
        '15_anos': 3780,
        '20_anos': 5040
    }
    
    def __init__(self, model_type='random_forest'):
        """
        Inicializa el predictor
        
        Args:
            model_type: 'random_forest' o 'gradient_boosting'
        """
        self.model_type = model_type
        self.models = {}  # Diccionario de modelos por horizonte
        self.feature_cols = None
        self.is_trained = False
        self.training_stats = {}  # Estadísticas de entrenamiento
        
    def _create_model(self):
        """Crea una instancia del modelo seleccionado"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Balancear clases
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")
    
    def train_all_horizons(self, data, test_size=0.2, custom_horizons=None):
        """
        Entrena modelos para todos los horizontes definidos
        
        Args:
            data: DataFrame con datos históricos (debe incluir 'close')
            test_size: Proporción de datos para test
            custom_horizons: Dict personalizado de horizontes (opcional)
        """
        print("=" * 80)
        print("🚀 INICIANDO ENTRENAMIENTO DE MODELOS MULTI-HORIZONTE")
        print("=" * 80)
        
        # Usar horizontes personalizados o los predefinidos
        horizons = custom_horizons if custom_horizons else self.HORIZONS
        
        # Preparar datos
        data = data.copy()
        data.columns = data.columns.str.lower()
        
        if 'close' not in data.columns:
            raise ValueError("El DataFrame debe contener la columna 'close'")
        
        n_samples = len(data)
        print(f"\n📊 Total de muestras disponibles: {n_samples:,}")
        print(f"📅 Rango de fechas: {data.index[0]} → {data.index[-1]}")
        
        # Seleccionar solo features numéricas (excluir 'close' del target)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'close' in numeric_cols:
            numeric_cols.remove('close')
        
        self.feature_cols = numeric_cols
        print(f"🔧 Features utilizadas: {len(self.feature_cols)}")
        
        # Entrenar modelo para cada horizonte
        trained_count = 0
        skipped_count = 0
        
        for horizon_name, horizon_days in sorted(horizons.items(), key=lambda x: x[1]):
            print(f"\n{'─' * 80}")
            print(f"📆 Horizonte: {horizon_name.upper().replace('_', ' ')} ({horizon_days} días)")
            print(f"{'─' * 80}")
            
            # Ajustar horizonte automáticamente si es muy largo
            original_horizon = horizon_days
            if horizon_days >= n_samples - 20:
                max_possible = max(n_samples - 50, 50)  # Dejar margen pero asegurar mínimo 50
                print(f"⚠️  Horizonte original demasiado largo ({horizon_days} días)")
                print(f"   Ajustando a máximo posible: {max_possible} días")
                horizon_days = max_possible
                
                # Si aún así es muy poco, saltar
                if max_possible < 30:
                    print(f"❌ Datos insuficientes incluso con ajuste (mínimo 30 días requerido)")
                    skipped_count += 1
                    continue
            
            try:
                # Crear target: 1 si sube, 0 si baja
                df = data.copy()
                target_col = f"target_h{horizon_days}"
                
                # Verificar si hay suficientes datos para este horizonte
                max_available_future = len(df) - 1
                
                if horizon_days > max_available_future:
                    print(f"⚠️  Horizonte {horizon_days} días excede datos disponibles ({max_available_future} días)")
                    print(f"   Ajustando horizonte a {max_available_future} días")
                    adjusted_horizon = max_available_future
                else:
                    adjusted_horizon = horizon_days
                
                df[target_col] = (df['close'].shift(-adjusted_horizon) > df['close']).astype(int)
                
                # Eliminar filas con NaN en el target
                df_clean = df.dropna(subset=[target_col])
                
                # Ajustar mínimo de muestras según horizonte
                if horizon_days > 756:  # Más de 3 años
                    min_samples_required = max(10, int(len(df_clean) * 0.05))
                elif horizon_days > 252:  # Más de 1 año  
                    min_samples_required = max(15, int(len(df_clean) * 0.08))
                else:
                    min_samples_required = min(50, max(20, int(n_samples * 0.1)))
                
                if len(df_clean) < min_samples_required:
                    print(f"⚠️  Solo {len(df_clean)} muestras válidas (mínimo {min_samples_required} requerido)")
                    skipped_count += 1
                    continue
                
                # Preparar X e y
                X = df_clean[self.feature_cols]
                y = df_clean[target_col]
                
                # Verificar balance de clases
                class_dist = y.value_counts()
                print(f"📊 Distribución de clases:")
                print(f"   📉 Bajadas (0): {class_dist.get(0, 0):,} ({class_dist.get(0, 0)/len(y)*100:.1f}%)")
                print(f"   📈 Subidas (1): {class_dist.get(1, 0):,} ({class_dist.get(1, 0)/len(y)*100:.1f}%)")
                
                # Ajustar test_size según cantidad de datos
                if len(df_clean) < 50:
                    actual_test_size = 0.2
                    print(f"⚠️  Ajustando test_size a {actual_test_size:.1%} (muy pocas muestras)")
                elif len(df_clean) < 100:
                    actual_test_size = min(test_size, 0.25)
                    print(f"⚠️  Ajustando test_size a {actual_test_size:.1%} (pocas muestras)")
                else:
                    actual_test_size = test_size
                
                # Split temporal (sin shuffle para respetar orden temporal)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=actual_test_size, random_state=42, shuffle=False
                )
                
                # Verificar que hay suficientes muestras para entrenar
                if len(X_train) < 10:
                    print(f"❌ Muy pocas muestras de entrenamiento ({len(X_train)}). Mínimo 10 requerido.")
                    skipped_count += 1
                    continue
                
                # Crear y entrenar modelo
                model = self._create_model()
                model.fit(X_train, y_train)
                
                # Evaluar
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                train_acc = accuracy_score(y_train, y_pred_train)
                test_acc = accuracy_score(y_test, y_pred_test)
                
                # Cross-validation solo si hay suficientes muestras
                if len(X_train) >= 50:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)//10), scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                else:
                    cv_mean = train_acc  # Usar train_acc si no hay suficientes datos para CV
                    cv_std = 0.0
                    print(f"⚠️  Omitiendo CV debido a pocas muestras")
                
                print(f"\n📈 Resultados del entrenamiento:")
                print(f"   Train Accuracy: {train_acc:.2%}")
                print(f"   Test Accuracy:  {test_acc:.2%}")
                if len(X_train) >= 50:
                    print(f"   CV Score (5-fold): {cv_mean:.2%} (±{cv_std:.2%})")
                print(f"   Samples Train: {len(X_train):,}")
                print(f"   Samples Test:  {len(X_test):,}")
                if original_horizon != horizon_days:
                    print(f"   ℹ️  Horizonte ajustado: {original_horizon} → {horizon_days} días")
                
                # Guardar modelo y estadísticas
                self.models[horizon_days] = {
                    'model': model,
                    'horizon_name': horizon_name,
                    'horizon_days': horizon_days,
                    'original_horizon': original_horizon,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'cv_score': cv_mean,
                    'cv_std': cv_std,
                    'samples': len(df_clean)
                }
                
                trained_count += 1
                print(f"✅ Modelo entrenado exitosamente")
                
            except Exception as e:
                print(f"❌ Error entrenando horizonte {horizon_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                skipped_count += 1
                continue
        
        print(f"\n{'=' * 80}")
        print(f"✅ ENTRENAMIENTO COMPLETADO")
        print(f"   Modelos entrenados: {trained_count}")
        print(f"   Horizontes omitidos: {skipped_count}")
        print(f"\n{'=' * 80}\n")
        
        if trained_count == 0:
            print("❌ ERROR: No se pudo entrenar ningún modelo")
            print(f"\n📊 Información de diagnóstico:")
            print(f"   • Total de muestras: {n_samples:,}")
            print(f"   • Features disponibles: {len(self.feature_cols)}")
            print(f"   • Horizontes intentados: {len(horizons)}")
            print(f"\n💡 Sugerencias:")
            print("   1. Descarga más datos históricos (period='max')")
            print("   2. Verifica que los datos tengan suficientes registros")
            print("   3. Revisa que las features no contengan todos NaN")
            raise Exception(f"❌ No se pudo entrenar ningún modelo. Se necesitan al menos 50 registros, tienes {n_samples}.")
        
        self.is_trained = True
        return trained_count
    
    def _calculate_historical_changes(self, data):
        """
        Calcula los cambios históricos promedio para cada horizonte
        
        Args:
            data: DataFrame con datos históricos
            
        Returns:
            Dict con cambios promedio por horizonte
        """
        historical_changes = {}
        
        for horizon_days in self.models.keys():
            # Calcular retornos históricos para este horizonte
            returns = (data['close'].shift(-horizon_days) / data['close'] - 1).dropna()
            
            # Separar subidas y bajadas
            up_returns = returns[returns > 0]
            down_returns = returns[returns < 0]
            
            historical_changes[horizon_days] = {
                'avg_up': up_returns.mean() if len(up_returns) > 0 else 0.0,
                'avg_down': down_returns.mean() if len(down_returns) > 0 else 0.0,
                'median_up': up_returns.median() if len(up_returns) > 0 else 0.0,
                'median_down': down_returns.median() if len(down_returns) > 0 else 0.0,
            }
        
        return historical_changes
    
    def predict_investment_horizons(self, data_with_features, show_details=True):
        """
        Predice si vale la pena invertir en cada horizonte temporal
        
        Args:
            data_with_features: DataFrame con features calculadas
            show_details: Si mostrar detalles adicionales
            
        Returns:
            DataFrame con predicciones por horizonte
        """
        if not self.is_trained:
            raise Exception("❌ El modelo no está entrenado. Ejecuta train_all_horizons() primero.")
        
        if self.feature_cols is None:
            raise Exception("❌ No hay features definidas.")
        
        # Calcular cambios históricos para cada horizonte
        historical_changes = self._calculate_historical_changes(data_with_features)
        
        # Obtener última fila con features
        last_row = data_with_features[self.feature_cols].iloc[-1:].copy()
        current_price = data_with_features['close'].iloc[-1]
        current_date = data_with_features.index[-1] if hasattr(data_with_features.index, 'date') else datetime.now()
        
        print("=" * 80)
        print("🔮 ANÁLISIS DE HORIZONTES DE INVERSIÓN")
        print("=" * 80)
        print(f"📅 Fecha de análisis: {current_date}")
        print(f"💰 Precio actual: ${current_price:.2f}")
        print(f"📊 Features utilizadas: {len(self.feature_cols)}")
        print("=" * 80)
        
        predictions = []
        
        for horizon_days in sorted(self.models.keys()):
            model_data = self.models[horizon_days]
            model = model_data['model']
            horizon_name = model_data['horizon_name']
            
            # Obtener cambios históricos para este horizonte específico
            hist_changes = historical_changes[horizon_days]
            
            # Hacer predicción
            pred_class = model.predict(last_row)[0]
            pred_proba = model.predict_proba(last_row)[0]
            
            # Manejar caso donde el modelo solo predijo una clase durante entrenamiento
            if len(pred_proba) == 1:
                # Solo hay una clase en el modelo
                if pred_class == 0:
                    prob_down = pred_proba[0]
                    prob_up = 0.0
                else:
                    prob_down = 0.0
                    prob_up = pred_proba[0]
            else:
                # Caso normal con ambas clases
                prob_down = pred_proba[0]
                prob_up = pred_proba[1]
            
            confidence = max(prob_down, prob_up)
            
            # Calcular cambio esperado ponderado por probabilidad
            # Usar cambios históricos promedio para cada dirección
            if pred_class == 1:  # Predicción de subida
                # Usar promedio de cambios históricos de subidas
                expected_change_pct = hist_changes['avg_up'] * prob_up + hist_changes['avg_down'] * prob_down
                signal = "📈 COMPRAR"
                direction = "ALCISTA"
            else:  # Predicción de bajada
                # Usar promedio de cambios históricos de bajadas
                expected_change_pct = hist_changes['avg_down'] * prob_down + hist_changes['avg_up'] * prob_up
                signal = "📉 VENDER"
                direction = "BAJISTA"
            
            # Calcular precio predicho
            predicted_price = current_price * (1 + expected_change_pct)
            change_amount = predicted_price - current_price
            
            # Nivel de confianza
            if confidence >= 0.75:
                conf_level = "🟢 ALTA"
            elif confidence >= 0.60:
                conf_level = "🟡 MEDIA"
            else:
                conf_level = "🔴 BAJA"
            
            predictions.append({
                'Horizonte': horizon_name.replace('_', ' ').title(),
                'Días': horizon_days,
                'Señal': signal,
                'Dirección': direction,
                'Precio Actual': current_price,
                'Precio Predicho': predicted_price,
                'Cambio $': change_amount,
                'Cambio %': expected_change_pct * 100,
                'Prob. Subida': prob_up * 100,
                'Prob. Bajada': prob_down * 100,
                'Confianza': confidence * 100,
                'Nivel': conf_level,
                'Test Acc': model_data['test_acc'] * 100
            })
        
        df_predictions = pd.DataFrame(predictions)
        
        # Mostrar resultados
        if show_details:
            self._print_pretty_predictions(df_predictions)
        
        return df_predictions
    
    def _print_pretty_predictions(self, df, current_price):
        """Imprime las predicciones de forma visual y clara"""
        print("\n📊 RECOMENDACIONES DE INVERSIÓN POR HORIZONTE TEMPORAL")
        print("=" * 150)
        
        # Formatear la tabla para mostrar
        display_df = df.copy()
        display_df['Precio Actual'] = display_df['Precio Actual'].apply(lambda x: f"${x:.2f}")
        display_df['Precio Predicho'] = display_df['Precio Predicho'].apply(lambda x: f"${x:.2f}")
        display_df['Cambio $'] = display_df['Cambio $'].apply(lambda x: f"${x:+.2f}")
        display_df['Cambio %'] = display_df['Cambio %'].apply(lambda x: f"{x:+.2f}%")
        display_df['Prob. Subida'] = display_df['Prob. Subida'].apply(lambda x: f"{x:.1f}%")
        display_df['Prob. Bajada'] = display_df['Prob. Bajada'].apply(lambda x: f"{x:.1f}%")
        display_df['Confianza'] = display_df['Confianza'].apply(lambda x: f"{x:.1f}%")
        display_df['Test Acc'] = display_df['Test Acc'].apply(lambda x: f"{x:.1f}%")
        
        # Formatear la tabla
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        pd.set_option('display.max_rows', None)
        
        print(display_df.to_string(index=False))
        
        print("\n" + "=" * 150)
        
        # Resumen por señal
        comprar = len(df[df['Señal'].str.contains('COMPRAR')])
        vender = len(df[df['Señal'].str.contains('VENDER')])
        
        print(f"\n📈 Señales COMPRAR: {comprar}/{len(df)} horizontes")
        print(f"📉 Señales VENDER:  {vender}/{len(df)} horizontes")
        
        # Mejor y peor escenario
        best_case = df.loc[df['Cambio %'].idxmax()]
        worst_case = df.loc[df['Cambio %'].idxmin()]
        
        print(f"\n🎯 Mejor escenario: {best_case['Horizonte']} ({best_case['Días']} días)")
        print(f"   Precio predicho: ${best_case['Precio Predicho']:.2f} ({best_case['Cambio %']:+.2f}%)")
        print(f"\n⚠️  Peor escenario: {worst_case['Horizonte']} ({worst_case['Días']} días)")
        print(f"   Precio predicho: ${worst_case['Precio Predicho']:.2f} ({worst_case['Cambio %']:+.2f}%)")
        
        # Recomendación general
        if comprar > vender:
            print(f"\n💡 RECOMENDACIÓN GENERAL: Momento favorable para INVERTIR")
        elif vender > comprar:
            print(f"\n💡 RECOMENDACIÓN GENERAL: Momento de CAUTELA o salida")
        else:
            print(f"\n💡 RECOMENDACIÓN GENERAL: Mercado NEUTRAL - Evaluar caso por caso")
        
        print("=" * 150)
    
    def get_feature_importance(self, horizon_days=None, top_n=10):
        """
        Obtiene la importancia de features para un horizonte específico
        
        Args:
            horizon_days: Horizonte específico (None = todos)
            top_n: Número de features principales a mostrar
        """
        if not self.is_trained:
            raise Exception("❌ El modelo no está entrenado.")
        
        if horizon_days is None:
            # Mostrar para todos los horizontes
            for h_days in sorted(self.models.keys()):
                self._print_feature_importance(h_days, top_n)
        else:
            if horizon_days not in self.models:
                raise ValueError(f"Horizonte {horizon_days} no encontrado en modelos entrenados")
            self._print_feature_importance(horizon_days, top_n)
    
    def _print_feature_importance(self, horizon_days, top_n):
        """Imprime la importancia de features para un horizonte"""
        model_data = self.models[horizon_days]
        model = model_data['model']
        horizon_name = model_data['horizon_name']
        
        if not hasattr(model, 'feature_importances_'):
            print(f"⚠️  Modelo {horizon_name} no soporta feature importance")
            return
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_cols,
            'Importancia': model.feature_importances_
        }).sort_values('Importancia', ascending=False).head(top_n)
        
        print(f"\n🔍 Top {top_n} Features - {horizon_name.replace('_', ' ').title()} ({horizon_days} días)")
        print("─" * 60)
        print(importance_df.to_string(index=False))
    
    def save_models(self, filepath='models/saved/improved_predictor.pkl', ticker=None, data_hash=None):
        """
        Guarda todos los modelos entrenados con metadatos
        
        Args:
            filepath: Ruta donde guardar los modelos
            ticker: Símbolo del ticker (ej: 'URTH', '^GSPC')
            data_hash: Hash de los datos usados para entrenar
        """
        print(f"\n💾 Guardando modelos en: {filepath}")
        print(f"   Ticker: {ticker}")
        print(f"   Modelos entrenados: {len(self.models)}")
        
        # Crear directorio si no existe
        dir_path = os.path.dirname(filepath)
        if not os.path.exists(dir_path):
            print(f"   Creando directorio: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
        
        save_data = {
            'models': self.models,
            'feature_cols': self.feature_cols,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'trained_date': datetime.now(),
            'horizons': self.HORIZONS,
            'ticker': ticker,
            'data_hash': data_hash,
            'n_samples': len(self.models[list(self.models.keys())[0]]['samples']) if self.models else 0
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            # Verificar que se guardó correctamente
            file_size = os.path.getsize(filepath)
            print(f"✅ Modelos guardados exitosamente")
            print(f"   Archivo: {filepath}")
            print(f"   Tamaño: {file_size / 1024 / 1024:.2f} MB")
            print(f"   Total de modelos: {len(self.models)}")
            
        except Exception as e:
            print(f"❌ ERROR guardando modelos: {e}")
            raise
    
    def load_models(self, filepath='models/saved/improved_predictor.pkl', verbose=True):
        """
        Carga modelos guardados
        
        Args:
            filepath: Ruta del archivo de modelos
            verbose: Si mostrar información de carga
            
        Returns:
            dict: Metadatos del modelo cargado
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.models = save_data['models']
        self.feature_cols = save_data['feature_cols']
        self.model_type = save_data['model_type']
        self.is_trained = save_data['is_trained']
        
        if verbose:
            print(f"📂 Modelos cargados desde: {filepath}")
            print(f"   Fecha de entrenamiento: {save_data['trained_date']}")
            print(f"   Ticker: {save_data.get('ticker', 'N/A')}")
            print(f"   Total de modelos: {len(self.models)}")
            print(f"   Horizontes disponibles: {list(save_data['horizons'].keys())}")
        
        return {
            'trained_date': save_data['trained_date'],
            'ticker': save_data.get('ticker'),
            'data_hash': save_data.get('data_hash'),
            'n_models': len(self.models)
        }
    
    @staticmethod
    def get_model_cache_path(ticker, model_type='random_forest', cache_dir='models/cache'):
        """
        Genera la ruta del archivo de caché para un ticker específico
        
        Args:
            ticker: Símbolo del ticker
            model_type: Tipo de modelo
            cache_dir: Directorio de caché
            
        Returns:
            str: Ruta del archivo de caché
        """
        # Limpiar el ticker para usar como nombre de archivo
        safe_ticker = ticker.replace('^', '').replace('.', '_')
        filename = f"{safe_ticker}_{model_type}.pkl"
        return os.path.join(cache_dir, filename)
    
    @staticmethod
    def check_cache_validity(filepath, max_age_days=7):
        """
        Verifica si un modelo en caché es válido
        
        Args:
            filepath: Ruta del archivo de caché
            max_age_days: Edad máxima en días (None = sin límite)
            
        Returns:
            tuple: (is_valid, metadata_dict)
        """
        if not os.path.exists(filepath):
            return False, None
        
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            trained_date = save_data.get('trained_date')
            if trained_date is None:
                return False, None
            
            # Verificar edad del modelo
            if max_age_days is not None:
                age_days = (datetime.now() - trained_date).days
                if age_days > max_age_days:
                    return False, {
                        'reason': 'expired',
                        'age_days': age_days,
                        'max_age_days': max_age_days
                    }
            
            return True, {
                'trained_date': trained_date,
                'ticker': save_data.get('ticker'),
                'n_models': len(save_data.get('models', {})),
                'age_days': (datetime.now() - trained_date).days
            }
            
        except Exception as e:
            return False, {'reason': 'error', 'error': str(e)}


def main():
    """Ejemplo de uso del sistema mejorado"""
    print("=" * 80)
    print("🔮 SISTEMA MEJORADO DE PREDICCIONES MULTI-HORIZONTE")
    print("=" * 80)
    
    # Nota: Descomentar cuando tengas los módulos disponibles
    from data.collectors.market_data import MarketDataCollector
    from data.processors.feature_engineering import FeatureEngineer
    
    # 1. Descargar datos históricos (máximo posible para 20 años)
    print("\n1️⃣ Descargando datos históricos...")
    collector = MarketDataCollector()
    data = collector.download_ticker('^GSPC', period='max')  # Máximo histórico
    print(f"   ✅ Descargados {len(data):,} registros")
    
    # 2. Crear features técnicos
    print("\n2️⃣ Creando features técnicos...")
    engineer = FeatureEngineer()
    data = engineer.add_technical_indicators(data)
    data = data.dropna()  # Eliminar NaN de indicadores
    print(f"   ✅ Dataset con {data.shape[1]} features")
    
    # 3. Entrenar modelos para todos los horizontes
    print("\n3️⃣ Entrenando modelos multi-horizonte...")
    predictor = ImprovedStockPredictor(model_type='random_forest')
    predictor.train_all_horizons(data, test_size=0.2)
    
    # 4. Hacer predicciones para todos los horizontes
    print("\n4️⃣ Generando predicciones...")
    predictions = predictor.predict_investment_horizons(data, show_details=True)
    
    # 5. Ver importancia de features (ejemplo para 1 año)
    print("\n5️⃣ Importancia de features...")
    predictor.get_feature_importance(horizon_days=252, top_n=10)
    
    # 6. Guardar modelos
    print("\n6️⃣ Guardando modelos...")
    predictor.save_models()
    
    print("\n" + "=" * 80)
    print("✅ SISTEMA COMPLETADO EXITOSAMENTE")
    print("=" * 80)
    
    print("\n⚠️  Para ejecutar el ejemplo completo, asegúrate de tener:")
    print("   • MarketDataCollector configurado")
    print("   • FeatureEngineer con indicadores técnicos")
    print("   • Datos históricos suficientes (idealmente 20+ años)")


if __name__ == "__main__":
    main()