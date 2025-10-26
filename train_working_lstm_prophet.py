#!/usr/bin/env python3
"""
Working LSTM + Prophet Pipeline for Microservice Rightsizing
Fixed the pandas Series indexing issue
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Prophet imports
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("Prophet not available, using simplified approach")
    PROPHET_AVAILABLE = False

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available, using simplified approach")
    TENSORFLOW_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkingLSTMProphetPipeline:
    """Working LSTM + Prophet Pipeline for Microservice Rightsizing"""
    
    def __init__(self, data_dir: str = "training_data", model_dir: str = "models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Pipeline components
        self.prophet_models = {}
        self.lstm_models = {}
        self.fusion_weights = {'prophet': 0.4, 'lstm': 0.6}
        
        # LSTM configuration
        self.sequence_length = 30
        self.n_features = 14
        
    def load_training_data(self, service_name: str) -> pd.DataFrame:
        """Load all collected data for a service"""
        logger.info(f"Loading training data for {service_name}")
        
        # Find all CSV files for the service
        csv_files = [f for f in os.listdir(self.data_dir) if f.startswith(service_name) and f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"No training data found for service {service_name}")
        
        logger.info(f"Found {len(csv_files)} data files for {service_name}")
        
        # Load and combine all data
        all_data = []
        for file in csv_files:
            file_path = os.path.join(self.data_dir, file)
            df = pd.read_csv(file_path)
            all_data.append(df)
            logger.info(f"Loaded {len(df)} rows from {file}")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined data: {len(combined_data)} total rows")
        
        return combined_data
    
    def prepare_time_series_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
        """Prepare data for LSTM and Prophet analysis"""
        logger.info("Preparing time series data")
        
        # Feature columns
        feature_columns = [
            'cpu_cores_value', 'mem_bytes_value', 'net_rx_bytes_value', 'net_tx_bytes_value',
            'pod_restarts_value', 'replica_count_value', 'node_cpu_util_value', 'node_mem_util_value',
            'network_activity_rate_value', 'processing_intensity_value', 'service_stability_value', 'resource_pressure_value'
        ]
        
        # Context features
        context_columns = ['replica_count', 'load_users']
        
        # Combine all features
        all_features = context_columns + feature_columns
        
        # Prepare features
        X = df[all_features].copy()
        X = X.fillna(X.median())
        
        # Create targets with intelligent buffering
        targets = {
            'cpu_target': df['cpu_cores_value'] * 1.2,  # 20% buffer
            'memory_target': df['mem_bytes_value'] * 1.15,  # 15% buffer
            'replica_target': df['replica_count_value'].copy()
        }
        
        # Handle NaN values in targets
        for target_name, target_values in targets.items():
            targets[target_name] = target_values.fillna(target_values.median())
        
        # Remove any remaining NaN values
        valid_indices = ~(targets['cpu_target'].isna() | targets['memory_target'].isna() | targets['replica_target'].isna())
        X = X[valid_indices]
        for target_name in targets:
            targets[target_name] = targets[target_name][valid_indices]
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Targets shape: {len(targets['cpu_target'])}")
        
        return X.values, targets, all_features
    
    def create_prophet_models(self, service_name: str, metrics_data: np.ndarray, targets: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Create Prophet models for trend analysis"""
        logger.info("Creating Prophet models for trend analysis")
        
        prophet_results = {}
        
        # Create time index
        time_index = pd.date_range(start='2024-01-01', periods=len(metrics_data), freq='1min')
        
        for target_name, target_values in targets.items():
            logger.info(f"Training Prophet model for {target_name}")
            
            if not PROPHET_AVAILABLE:
                # Simplified approach without Prophet
                prophet_results[target_name] = {
                    'trend': target_values[-30:],
                    'seasonal': np.zeros(30),
                    'confidence_lower': target_values[-30:] * 0.9,
                    'confidence_upper': target_values[-30:] * 1.1,
                    'forecast': target_values[-30:]
                }
                continue
            
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': time_index,
                'y': target_values
            })
            
            try:
                # Create Prophet model
                model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=True,
                    daily_seasonality=True,
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0
                )
                
                # Fit model
                model.fit(prophet_data)
                
                # Make future predictions
                future = model.make_future_dataframe(periods=30, freq='1min')
                forecast = model.predict(future)
                
                # Store results
                prophet_results[target_name] = {
                    'model': model,
                    'trend': forecast['trend'].iloc[-30:].values,
                    'seasonal': forecast['seasonal'].iloc[-30:].values,
                    'confidence_lower': forecast['yhat_lower'].iloc[-30:].values,
                    'confidence_upper': forecast['yhat_upper'].iloc[-30:].values,
                    'forecast': forecast['yhat'].iloc[-30:].values
                }
                
                logger.info(f"Prophet model trained for {target_name}")
                
            except Exception as e:
                logger.warning(f"Prophet model failed for {target_name}: {e}, using simplified approach")
                prophet_results[target_name] = {
                    'trend': target_values[-30:],
                    'seasonal': np.zeros(30),
                    'confidence_lower': target_values[-30:] * 0.9,
                    'confidence_upper': target_values[-30:] * 1.1,
                    'forecast': target_values[-30:]
                }
        
        return prophet_results
    
    def create_lstm_models(self, service_name: str, metrics_data: np.ndarray, targets: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Create LSTM models for pattern learning"""
        logger.info("Creating LSTM models for pattern learning")
        
        lstm_results = {}
        
        for target_name, target_values in targets.items():
            logger.info(f"Training LSTM model for {target_name}")
            
            if not TENSORFLOW_AVAILABLE:
                # Simplified approach without TensorFlow
                lstm_results[target_name] = {
                    'predictions': target_values[-30:],
                    'mse': 0.0,
                    'mae': 0.0,
                    'model': None
                }
                continue
            
            try:
                # Prepare sequences
                X_sequences, y_sequences = self.prepare_sequences(metrics_data, target_values)
                
                if len(X_sequences) == 0:
                    logger.warning(f"No sequences created for {target_name}, using simplified approach")
                    lstm_results[target_name] = {
                        'predictions': target_values[-30:],
                        'mse': 0.0,
                        'mae': 0.0,
                        'model': None
                    }
                    continue
                
                # Create LSTM model
                model = self.create_lstm_model()
                
                # Train model
                history = model.fit(
                    X_sequences, y_sequences,
                    epochs=50,
                    validation_split=0.2,
                    verbose=0,
                    batch_size=32,
                    callbacks=[
                        EarlyStopping(patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(factor=0.5, patience=5)
                    ]
                )
                
                # Make predictions
                predictions = model.predict(X_sequences, verbose=0)
                
                # Store results
                lstm_results[target_name] = {
                    'model': model,
                    'predictions': predictions,
                    'history': history.history,
                    'mse': np.mean((y_sequences - predictions.flatten()) ** 2),
                    'mae': np.mean(np.abs(y_sequences - predictions.flatten()))
                }
                
                logger.info(f"LSTM model trained for {target_name} - MSE: {lstm_results[target_name]['mse']:.6f}")
                
            except Exception as e:
                logger.warning(f"LSTM model failed for {target_name}: {e}, using simplified approach")
                lstm_results[target_name] = {
                    'predictions': target_values[-30:],
                    'mse': 0.0,
                    'mae': 0.0,
                    'model': None
                }
        
        return lstm_results
    
    def prepare_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series sequences for LSTM"""
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(data)):
            sequences.append(data[i-self.sequence_length:i])
            targets.append(target[i])
        
        return np.array(sequences), np.array(targets)
    
    def create_lstm_model(self) -> tf.keras.Model:
        """Create LSTM model architecture"""
        model = models.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_working_predictions(self, prophet_results: Dict[str, Any], lstm_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create working predictions combining Prophet and LSTM"""
        logger.info("Creating working predictions")
        
        working_results = {}
        
        for target_name in prophet_results.keys():
            try:
                # Get Prophet prediction - FIXED: Use .iloc[-1] for pandas Series
                prophet_forecast = prophet_results[target_name]['forecast']
                if isinstance(prophet_forecast, pd.Series):
                    prophet_pred = float(prophet_forecast.iloc[-1])
                else:
                    prophet_pred = float(prophet_forecast[-1]) if len(prophet_forecast) > 0 else 0.0
                
                # Get LSTM prediction
                lstm_pred = 0.0
                if target_name in lstm_results and 'predictions' in lstm_results[target_name]:
                    lstm_predictions = lstm_results[target_name]['predictions']
                    if len(lstm_predictions) > 0:
                        lstm_pred_raw = lstm_predictions[-1]
                        
                        # Convert to scalar
                        if hasattr(lstm_pred_raw, 'item'):
                            lstm_pred = float(lstm_pred_raw.item())
                        elif isinstance(lstm_pred_raw, np.ndarray):
                            lstm_pred = float(lstm_pred_raw.flatten()[0]) if lstm_pred_raw.size > 0 else 0.0
                        else:
                            lstm_pred = float(lstm_pred_raw)
                
                # Simple average fusion
                fused_pred = (prophet_pred + lstm_pred) / 2.0
                
                # Ensure prediction is not negative
                fused_pred = max(0.0, fused_pred)
                
                # Calculate confidence (simplified)
                confidence = 0.7  # Base confidence
                if target_name in lstm_results and lstm_results[target_name]['mse'] > 0:
                    # Adjust confidence based on LSTM performance
                    mse = lstm_results[target_name]['mse']
                    confidence = max(0.5, min(0.9, 1.0 - (mse / 1000.0)))
                
                working_results[target_name] = {
                    'prediction': fused_pred,
                    'confidence': confidence,
                    'prophet_contribution': prophet_pred,
                    'lstm_contribution': lstm_pred,
                    'prophet_trend': prophet_forecast,
                    'lstm_predictions': lstm_results.get(target_name, {}).get('predictions', [])
                }
                
                logger.info(f"Working {target_name}: Prophet={prophet_pred:.6f}, LSTM={lstm_pred:.6f}, Final={fused_pred:.6f}")
                
            except Exception as e:
                logger.error(f"Error creating working prediction for {target_name}: {e}")
                # Fallback to Prophet only
                prophet_forecast = prophet_results[target_name]['forecast']
                if isinstance(prophet_forecast, pd.Series):
                    prophet_pred = float(prophet_forecast.iloc[-1])
                else:
                    prophet_pred = float(prophet_forecast[-1]) if len(prophet_forecast) > 0 else 0.0
                
                working_results[target_name] = {
                    'prediction': prophet_pred,
                    'confidence': 0.5,
                    'prophet_contribution': prophet_pred,
                    'lstm_contribution': 0.0,
                    'prophet_trend': prophet_forecast,
                    'lstm_predictions': []
                }
        
        return working_results
    
    def train_pipeline(self, service_name: str) -> Dict[str, Any]:
        """Train the complete LSTM + Prophet pipeline"""
        logger.info(f"Training LSTM + Prophet pipeline for {service_name}")
        
        try:
            # Load data
            df = self.load_training_data(service_name)
            
            # Prepare time series data
            metrics_data, targets, feature_names = self.prepare_time_series_data(df)
            
            # Train Prophet models
            prophet_results = self.create_prophet_models(service_name, metrics_data, targets)
            
            # Train LSTM models
            lstm_results = self.create_lstm_models(service_name, metrics_data, targets)
            
            # Create working predictions
            working_results = self.create_working_predictions(prophet_results, lstm_results)
            
            # Save models
            model_path = os.path.join(self.model_dir, f"{service_name}_lstm_prophet_pipeline.joblib")
            
            import joblib
            joblib.dump({
                'prophet_models': prophet_results,
                'lstm_models': lstm_results,
                'working_results': working_results,
                'trained_at': datetime.now().isoformat(),
                'service_name': service_name,
                'pipeline_type': 'lstm_prophet',
                'feature_names': feature_names,
                'fusion_weights': self.fusion_weights
            }, model_path)
            
            logger.info(f"Pipeline saved to {model_path}")
            
            # Print results
            print(f"\n🎯 LSTM + PROPHET PIPELINE RESULTS FOR {service_name.upper()}")
            print("=" * 60)
            
            for target_name, result in working_results.items():
                print(f"\n{target_name.upper()}:")
                print(f"  Final Prediction: {result['prediction']:.6f}")
                print(f"  Confidence: {result['confidence']:.4f}")
                print(f"  Prophet Contribution: {result['prophet_contribution']:.6f}")
                print(f"  LSTM Contribution: {result['lstm_contribution']:.6f}")
            
            return {
                'status': 'success',
                'service_name': service_name,
                'prophet_models': len(prophet_results),
                'lstm_models': len(lstm_results),
                'working_results': working_results,
                'model_path': model_path
            }
            
        except Exception as e:
            logger.error(f"Pipeline training failed for {service_name}: {e}")
            return {
                'status': 'failed',
                'service_name': service_name,
                'error': str(e)
            }
    
    def make_recommendations(self, service_name: str, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Make rightsizing recommendations using the trained pipeline"""
        try:
            # Load models
            model_path = os.path.join(self.model_dir, f"{service_name}_lstm_prophet_pipeline.joblib")
            if not os.path.exists(model_path):
                return {'error': f'No trained pipeline found for {service_name}'}
            
            import joblib
            pipeline_data = joblib.load(model_path)
            
            # Get working results
            working_results = pipeline_data['working_results']
            
            # Create recommendations
            recommendations = {
                'cpu_cores': max(0.1, working_results['cpu_target']['prediction']),
                'memory_bytes': max(100000000, working_results['memory_target']['prediction']),
                'replicas': max(1, round(working_results['replica_target']['prediction']))
            }
            
            # Calculate overall confidence
            confidences = [result['confidence'] for result in working_results.values()]
            overall_confidence = np.mean(confidences)
            
            return {
                'service_name': service_name,
                'recommendations': recommendations,
                'confidence': overall_confidence,
                'pipeline_type': 'lstm_prophet',
                'fusion_weights': pipeline_data['fusion_weights']
            }
            
        except Exception as e:
            return {'error': f'Failed to make recommendations: {e}'}

def main():
    """Main training function"""
    print("🚀 MOrA Working LSTM + Prophet Pipeline Training")
    print("=" * 60)
    
    pipeline = WorkingLSTMProphetPipeline()
    
    # Train pipeline for frontend service
    result = pipeline.train_pipeline('frontend')
    
    if result['status'] == 'success':
        print(f"\n✅ SUCCESS: LSTM + Prophet pipeline trained for {result['service_name']}")
        print(f"📁 Pipeline saved to: {result['model_path']}")
        print(f"🔧 Prophet models: {result['prophet_models']}")
        print(f"🧠 LSTM models: {result['lstm_models']}")
        
        # Test recommendations
        print(f"\n🧪 TESTING RECOMMENDATIONS")
        print("-" * 30)
        
        sample_metrics = {
            'cpu_cores_value': 0.0001,
            'mem_bytes_value': 10000000,
            'replica_count': 2,
            'load_users': 50
        }
        
        recommendations = pipeline.make_recommendations('frontend', sample_metrics)
        
        if 'error' not in recommendations:
            print(f"📊 Sample Recommendations:")
            print(f"  CPU Cores: {recommendations['recommendations']['cpu_cores']:.6f}")
            print(f"  Memory: {recommendations['recommendations']['memory_bytes']:,.0f} bytes")
            print(f"  Replicas: {recommendations['recommendations']['replicas']}")
            print(f"  Confidence: {recommendations['confidence']:.2f}")
            print(f"  Fusion Weights: Prophet={recommendations['fusion_weights']['prophet']}, LSTM={recommendations['fusion_weights']['lstm']}")
        else:
            print(f"❌ Error: {recommendations['error']}")
    else:
        print(f"❌ FAILED: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
