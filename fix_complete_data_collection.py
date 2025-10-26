#!/usr/bin/env python3
"""
Fix Complete Data Collection
Ensures all scenarios and configurations are collected
"""
import sys
import os
import time
import logging
sys.path.append('.')

from src.mora.core.data_acquisition import DataAcquisitionPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collect_complete_frontend_data():
    """Collect complete frontend data with all scenarios and configurations"""
    
    logger.info("🚀 COMPLETE FRONTEND DATA COLLECTION")
    logger.info("=" * 50)
    
    # Create pipeline
    pipeline = DataAcquisitionPipeline()
    
    # Complete configuration
    config = {
        "experiment_duration_minutes": 15,
        "sample_interval": "30s", 
        "replica_counts": [1, 2, 4],
        "load_levels_users": [5, 10, 20, 30, 50, 75],
        "test_scenarios": ["browsing", "checkout"],
        "stabilization_wait_seconds": 180
    }
    
    logger.info(f"Configuration:")
    logger.info(f"  - Scenarios: {config['test_scenarios']}")
    logger.info(f"  - Replicas: {config['replica_counts']}")
    logger.info(f"  - Load levels: {config['load_levels_users']}")
    logger.info(f"  - Duration: {config['experiment_duration_minutes']} minutes")
    
    total_expected = len(config['test_scenarios']) * len(config['replica_counts']) * len(config['load_levels_users'])
    logger.info(f"  - Total expected: {total_expected} experiments")
    
    try:
        # Run complete experiment
        logger.info("🔄 Starting complete data collection...")
        result = pipeline.run_isolated_training_experiment("frontend", config)
        
        # Check results
        import glob
        all_files = glob.glob("training_data/frontend_*.csv")
        browsing_files = glob.glob("training_data/frontend_browsing_*.csv")
        checkout_files = glob.glob("training_data/frontend_checkout_*.csv")
        
        logger.info("📊 COLLECTION RESULTS:")
        logger.info(f"  - Total files: {len(all_files)}")
        logger.info(f"  - Browsing files: {len(browsing_files)}")
        logger.info(f"  - Checkout files: {len(checkout_files)}")
        
        if len(browsing_files) > 0 and len(checkout_files) > 0:
            logger.info("✅ SUCCESS: Both scenarios collected!")
            return True
        else:
            logger.error("❌ ISSUE: Missing scenarios")
            if len(browsing_files) == 0:
                logger.error("❌ No browsing experiments")
            if len(checkout_files) == 0:
                logger.error("❌ No checkout experiments")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_frontend_model():
    """Train LSTM + Prophet model for frontend"""
    logger.info("🧠 TRAINING FRONTEND MODEL")
    logger.info("=" * 30)
    
    try:
        from train_working_lstm_prophet import WorkingLSTMProphetPipeline
        
        trainer = WorkingLSTMProphetPipeline()
        result = trainer.train_pipeline("frontend")
        
        if result.get("status") == "success":
            logger.info("✅ Model training successful!")
            return True
        else:
            logger.error("❌ Model training failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error training model: {e}")
        return False

def main():
    """Main execution"""
    print("🔧 FIXING COMPLETE DATA COLLECTION")
    print("=" * 50)
    
    # Step 1: Collect complete data
    print("Step 1: Collecting complete frontend data...")
    data_success = collect_complete_frontend_data()
    
    if not data_success:
        print("❌ Data collection failed. Stopping.")
        return
    
    # Step 2: Train model
    print("\nStep 2: Training frontend model...")
    model_success = train_frontend_model()
    
    if model_success:
        print("✅ Frontend service completed successfully!")
    else:
        print("❌ Model training failed.")

if __name__ == "__main__":
    main()
