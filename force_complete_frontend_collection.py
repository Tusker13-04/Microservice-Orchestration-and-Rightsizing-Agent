#!/usr/bin/env python3
"""
Force complete frontend data collection with both browsing and checkout scenarios
"""
import sys
import os
sys.path.append('.')

from src.mora.core.data_acquisition import DataAcquisitionPipeline

def force_complete_frontend_collection():
    """Force complete frontend data collection with both scenarios"""
    print("🚀 FORCING COMPLETE FRONTEND DATA COLLECTION")
    print("=" * 60)
    
    # Create pipeline
    pipeline = DataAcquisitionPipeline()
    
    # Configuration for complete frontend collection
    config = {
        "experiment_duration_minutes": 15,
        "sample_interval": "30s",
        "replica_counts": [1, 2, 4],
        "load_levels_users": [5, 10, 20, 30, 50, 75],
        "test_scenarios": ["browsing", "checkout"],  # Both scenarios
        "stabilization_wait_seconds": 180
    }
    
    print(f"Configuration:")
    print(f"  - Scenarios: {config['test_scenarios']}")
    print(f"  - Replicas: {config['replica_counts']}")
    print(f"  - Load levels: {config['load_levels_users']}")
    print(f"  - Duration: {config['experiment_duration_minutes']} minutes")
    print(f"  - Total experiments: {len(config['test_scenarios'])} × {len(config['replica_counts'])} × {len(config['load_levels_users'])} = {len(config['test_scenarios']) * len(config['replica_counts']) * len(config['load_levels_users'])}")
    print("")
    
    try:
        # Clear any existing data to force fresh collection
        print("🧹 Clearing existing data to force fresh collection...")
        import shutil
        if os.path.exists("training_data"):
            shutil.rmtree("training_data")
        os.makedirs("training_data", exist_ok=True)
        print("✅ Data cleared")
        print("")
        
        # Run complete experiment
        print("🔄 Starting complete frontend data collection...")
        result = pipeline.run_isolated_training_experiment("frontend", config)
        
        print("✅ Collection completed!")
        print("")
        
        # Check results
        import glob
        all_files = glob.glob("training_data/frontend_*.csv")
        browsing_files = glob.glob("training_data/frontend_browsing_*.csv")
        checkout_files = glob.glob("training_data/frontend_checkout_*.csv")
        
        print("📊 COLLECTION RESULTS:")
        print(f"  - Total files: {len(all_files)}")
        print(f"  - Browsing files: {len(browsing_files)}")
        print(f"  - Checkout files: {len(checkout_files)}")
        print("")
        
        if len(browsing_files) > 0 and len(checkout_files) > 0:
            print("✅ SUCCESS: Both scenarios collected!")
            print(f"✅ Browsing: {len(browsing_files)} experiments")
            print(f"✅ Checkout: {len(checkout_files)} experiments")
        else:
            print("❌ ISSUE: Missing scenarios")
            if len(browsing_files) == 0:
                print("❌ No browsing experiments")
            if len(checkout_files) == 0:
                print("❌ No checkout experiments")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    force_complete_frontend_collection()
