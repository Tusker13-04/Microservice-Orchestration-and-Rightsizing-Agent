#!/usr/bin/env python3
"""
Complete Service Training Script
Handles end-to-end training for all services with proper error handling
"""
import sys
import os
import time
import logging
from typing import List, Dict, Any
sys.path.append('.')

from src.mora.core.data_acquisition import DataAcquisitionPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteServiceTrainer:
    """Handles complete end-to-end training for all services"""
    
    def __init__(self):
        self.pipeline = DataAcquisitionPipeline()
        # Start with just frontend for testing
        self.services = [
            "frontend"
        ]
        
        # Resource-optimized configuration (flat structure for run_isolated_training_experiment)
        self.config = {
            "experiment_duration_minutes": 15,
            "sample_interval": "30s",
            "replica_counts": [1, 2, 4],
            "load_levels_users": [5, 10, 20, 30, 50, 75],
            "test_scenarios": ["browsing", "checkout"],
            "stabilization_wait_seconds": 180
        }
    
    def train_service(self, service_name: str) -> Dict[str, Any]:
        """Train a single service end-to-end"""
        logger.info(f"🚀 Starting complete training for {service_name}")
        
        try:
            # Step 1: Data Collection
            logger.info(f"📊 Collecting data for {service_name}...")
            result = self.pipeline.run_isolated_training_experiment(service_name, self.config)
            
            # Check if data was collected
            import glob
            data_files = glob.glob(f"training_data/{service_name}_*.csv")
            logger.info(f"✅ Collected {len(data_files)} experiments for {service_name}")
            
            if len(data_files) == 0:
                logger.error(f"❌ No data collected for {service_name}")
                return {"status": "failed", "error": "No data collected"}
            
            # Step 2: Model Training
            logger.info(f"🧠 Training LSTM + Prophet models for {service_name}...")
            from train_working_lstm_prophet import WorkingLSTMProphetPipeline
            
            trainer = WorkingLSTMProphetPipeline()
            training_result = trainer.train_pipeline(service_name)
            
            if training_result.get("status") == "success":
                logger.info(f"✅ Successfully trained models for {service_name}")
                return {
                    "status": "success",
                    "service": service_name,
                    "data_experiments": len(data_files),
                    "model_training": training_result
                }
            else:
                logger.error(f"❌ Model training failed for {service_name}")
                return {"status": "failed", "error": "Model training failed"}
                
        except Exception as e:
            logger.error(f"❌ Error training {service_name}: {e}")
            return {"status": "failed", "error": str(e)}
    
    def train_all_services(self) -> Dict[str, Any]:
        """Train all services sequentially"""
        logger.info("🎯 Starting complete training for all services")
        logger.info(f"Services to train: {self.services}")
        
        results = {}
        successful_services = []
        failed_services = []
        
        for i, service in enumerate(self.services, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 Processing Service {i}/{len(self.services)}: {service}")
            logger.info(f"{'='*60}")
            
            result = self.train_service(service)
            results[service] = result
            
            if result.get("status") == "success":
                successful_services.append(service)
                logger.info(f"✅ {service} completed successfully")
            else:
                failed_services.append(service)
                logger.error(f"❌ {service} failed: {result.get('error', 'Unknown error')}")
            
            # Brief pause between services
            if i < len(self.services):
                logger.info("⏳ Pausing 30 seconds before next service...")
                time.sleep(30)
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("📊 TRAINING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Successful: {len(successful_services)} services")
        logger.info(f"❌ Failed: {len(failed_services)} services")
        
        if successful_services:
            logger.info(f"✅ Successful services: {', '.join(successful_services)}")
        
        if failed_services:
            logger.info(f"❌ Failed services: {', '.join(failed_services)}")
        
        return {
            "status": "completed",
            "total_services": len(self.services),
            "successful": len(successful_services),
            "failed": len(failed_services),
            "successful_services": successful_services,
            "failed_services": failed_services,
            "results": results
        }

def main():
    """Main execution function"""
    print("🚀 COMPLETE SERVICE TRAINING SYSTEM")
    print("=" * 60)
    print("This will train all 10 services end-to-end")
    print("Estimated time: 15+ hours")
    print("=" * 60)
    
    trainer = CompleteServiceTrainer()
    results = trainer.train_all_services()
    
    print(f"\n🎯 FINAL RESULTS")
    print(f"✅ Successful: {results['successful']}/{results['total_services']}")
    print(f"❌ Failed: {results['failed']}/{results['total_services']}")
    
    if results['successful'] > 0:
        print(f"✅ Phase 2 Complete for {results['successful']} services!")
    
    return results

if __name__ == "__main__":
    main()
