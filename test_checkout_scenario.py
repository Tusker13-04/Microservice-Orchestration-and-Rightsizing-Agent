#!/usr/bin/env python3
"""
Test script to debug checkout scenario collection
"""
import sys
import os
sys.path.append('.')

from src.mora.core.data_acquisition import DataAcquisitionPipeline

def test_checkout_scenario():
    """Test checkout scenario collection specifically"""
    print("🔍 TESTING CHECKOUT SCENARIO COLLECTION")
    print("=" * 50)
    
    # Create pipeline
    pipeline = DataAcquisitionPipeline()
    
    # Test configuration with only checkout scenario
    config = {
        "experiment_duration_minutes": 5,  # Short test
        "sample_interval": "30s",
        "replica_counts": [1],  # Only 1 replica for test
        "load_levels_users": [5],  # Only 5 users for test
        "test_scenarios": ["checkout"],  # Only checkout scenario
        "stabilization_wait_seconds": 60
    }
    
    print(f"Configuration: {config}")
    print("")
    
    try:
        # Run experiment
        result = pipeline.run_isolated_training_experiment("frontend", config)
        
        print("✅ Experiment completed!")
        print(f"Result: {result}")
        
        # Check if checkout data was created
        import glob
        checkout_files = glob.glob("training_data/frontend_checkout_*.csv")
        print(f"Checkout files created: {len(checkout_files)}")
        for file in checkout_files:
            print(f"  - {file}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_checkout_scenario()
