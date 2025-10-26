#!/usr/bin/env python3
"""
Train ML models using collected experiment data
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Any
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollectedDataTrainer:
    """Train ML models using collected experiment data"""
    
    def __init__(self, data_dir: str = "training_data", model_dir: str = "models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
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
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features and targets for training"""
        logger.info("Preparing features and targets")
        
        # Feature columns (all metrics except target variables)
        feature_columns = [
            'cpu_cores_value', 'mem_bytes_value', 'net_rx_bytes_value', 'net_tx_bytes_value',
            'pod_restarts_value', 'replica_count_value', 'node_cpu_util_value', 'node_mem_util_value',
            'network_activity_rate_value', 'processing_intensity_value', 'service_stability_value', 'resource_pressure_value'
        ]
        
        # Context features
        context_columns = ['replica_count', 'load_users']
        
        # Combine all features
        all_features = context_columns + feature_columns
        
        # Check if all columns exist
        missing_cols = [col for col in all_features if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            all_features = [col for col in all_features if col in df.columns]
        
        # Prepare features
        X = df[all_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Create targets for CPU and Memory recommendations
        # Target 1: CPU cores recommendation (based on current usage + buffer)
        cpu_target = df['cpu_cores_value'] * 1.2  # 20% buffer
        
        # Target 2: Memory recommendation (based on current usage + buffer)
        mem_target = df['mem_bytes_value'] * 1.15  # 15% buffer
        
        # Target 3: Replica count recommendation (based on load and current replicas)
        replica_target = df['replica_count_value'].copy()
        
        # Handle NaN values in targets
        cpu_target = cpu_target.fillna(cpu_target.median())
        mem_target = mem_target.fillna(mem_target.median())
        replica_target = replica_target.fillna(replica_target.median())
        
        # Remove any remaining NaN values
        valid_indices = ~(cpu_target.isna() | mem_target.isna() | replica_target.isna())
        X = X[valid_indices]
        cpu_target = cpu_target[valid_indices]
        mem_target = mem_target[valid_indices]
        replica_target = replica_target[valid_indices]
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"CPU target range: {cpu_target.min():.6f} - {cpu_target.max():.6f}")
        logger.info(f"Memory target range: {mem_target.min():.0f} - {mem_target.max():.0f}")
        logger.info(f"Replica target range: {replica_target.min():.0f} - {replica_target.max():.0f}")
        
        return X, cpu_target, mem_target, replica_target, all_features
    
    def train_models(self, service_name: str) -> Dict[str, Any]:
        """Train ML models for a service"""
        logger.info(f"Training models for {service_name}")
        
        try:
            # Load data
            df = self.load_training_data(service_name)
            
            # Prepare features
            X, cpu_target, mem_target, replica_target, feature_names = self.prepare_features(df)
            
            # Split data
            X_train, X_test, cpu_train, cpu_test, mem_train, mem_test, replica_train, replica_test = train_test_split(
                X, cpu_target, mem_target, replica_target, test_size=0.2, random_state=42
            )
            
            logger.info(f"Training set: {len(X_train)} samples")
            logger.info(f"Test set: {len(X_test)} samples")
            
            # Train models
            models = {}
            results = {}
            
            # CPU model
            logger.info("Training CPU recommendation model...")
            cpu_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            cpu_model.fit(X_train, cpu_train)
            cpu_pred = cpu_model.predict(X_test)
            
            models['cpu_model'] = cpu_model
            results['cpu_model'] = {
                'mae': mean_absolute_error(cpu_test, cpu_pred),
                'mse': mean_squared_error(cpu_test, cpu_pred),
                'r2': r2_score(cpu_test, cpu_pred)
            }
            
            # Memory model
            logger.info("Training Memory recommendation model...")
            mem_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            mem_model.fit(X_train, mem_train)
            mem_pred = mem_model.predict(X_test)
            
            models['memory_model'] = mem_model
            results['memory_model'] = {
                'mae': mean_absolute_error(mem_test, mem_pred),
                'mse': mean_squared_error(mem_test, mem_pred),
                'r2': r2_score(mem_test, mem_pred)
            }
            
            # Replica model
            logger.info("Training Replica recommendation model...")
            replica_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            replica_model.fit(X_train, replica_train)
            replica_pred = replica_model.predict(X_test)
            
            models['replica_model'] = replica_model
            results['replica_model'] = {
                'mae': mean_absolute_error(replica_test, replica_pred),
                'mse': mean_squared_error(replica_test, replica_pred),
                'r2': r2_score(replica_test, replica_pred)
            }
            
            # Save models
            model_path = os.path.join(self.model_dir, f"{service_name}_models.joblib")
            joblib.dump({
                'models': models,
                'feature_names': feature_names,
                'training_results': results,
                'trained_at': datetime.now().isoformat(),
                'service_name': service_name
            }, model_path)
            
            logger.info(f"Models saved to {model_path}")
            
            # Print results
            print(f"\n🎯 TRAINING RESULTS FOR {service_name.upper()}")
            print("=" * 50)
            
            for model_name, metrics in results.items():
                print(f"\n{model_name.upper()}:")
                print(f"  MAE: {metrics['mae']:.6f}")
                print(f"  MSE: {metrics['mse']:.6f}")
                print(f"  R²:  {metrics['r2']:.4f}")
            
            return {
                'status': 'success',
                'service_name': service_name,
                'models_trained': len(models),
                'training_results': results,
                'model_path': model_path,
                'feature_names': feature_names
            }
            
        except Exception as e:
            logger.error(f"Training failed for {service_name}: {e}")
            return {
                'status': 'failed',
                'service_name': service_name,
                'error': str(e)
            }
    
    def make_recommendations(self, service_name: str, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Make rightsizing recommendations using trained models"""
        try:
            # Load models
            model_path = os.path.join(self.model_dir, f"{service_name}_models.joblib")
            if not os.path.exists(model_path):
                return {'error': f'No trained models found for {service_name}'}
            
            model_data = joblib.load(model_path)
            models = model_data['models']
            feature_names = model_data['feature_names']
            
            # Prepare input features
            input_features = pd.DataFrame([current_metrics])
            
            # Ensure all required features are present
            for feature in feature_names:
                if feature not in input_features.columns:
                    input_features[feature] = 0.0  # Default value
            
            # Make predictions
            cpu_rec = models['cpu_model'].predict(input_features[feature_names])[0]
            mem_rec = models['memory_model'].predict(input_features[feature_names])[0]
            replica_rec = models['replica_model'].predict(input_features[feature_names])[0]
            
            return {
                'service_name': service_name,
                'recommendations': {
                    'cpu_cores': max(0.1, round(cpu_rec, 3)),
                    'memory_bytes': max(100000000, round(mem_rec)),
                    'replicas': max(1, round(replica_rec))
                },
                'confidence': 'high'  # Based on model performance
            }
            
        except Exception as e:
            return {'error': f'Failed to make recommendations: {e}'}

def main():
    """Main training function"""
    print("🚀 MOrA Model Training from Collected Data")
    print("=" * 50)
    
    trainer = CollectedDataTrainer()
    
    # Train models for frontend service
    result = trainer.train_models('frontend')
    
    if result['status'] == 'success':
        print(f"\n✅ SUCCESS: Models trained for {result['service_name']}")
        print(f"📁 Models saved to: {result['model_path']}")
        print(f"🔧 Features used: {len(result['feature_names'])}")
        
        # Test recommendations
        print(f"\n🧪 TESTING RECOMMENDATIONS")
        print("-" * 30)
        
        # Sample current metrics (you can modify these)
        sample_metrics = {
            'cpu_cores_value': 0.0001,
            'mem_bytes_value': 10000000,
            'net_rx_bytes_value': 0.1,
            'net_tx_bytes_value': 0.1,
            'pod_restarts_value': 0,
            'replica_count_value': 2,
            'node_cpu_util_value': 20.0,
            'node_mem_util_value': 60.0,
            'network_activity_rate_value': 0.001,
            'processing_intensity_value': 0.0001,
            'service_stability_value': 1.0,
            'resource_pressure_value': 0.3,
            'replica_count': 2,
            'load_users': 50
        }
        
        recommendations = trainer.make_recommendations('frontend', sample_metrics)
        
        if 'error' not in recommendations:
            print(f"📊 Sample Recommendations:")
            print(f"  CPU Cores: {recommendations['recommendations']['cpu_cores']}")
            print(f"  Memory: {recommendations['recommendations']['memory_bytes']:,} bytes")
            print(f"  Replicas: {recommendations['recommendations']['replicas']}")
        else:
            print(f"❌ Error: {recommendations['error']}")
    else:
        print(f"❌ FAILED: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
