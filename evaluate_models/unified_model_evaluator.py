#!/usr/bin/env python3
"""
MOrA Unified Model Evaluation Suite
==================================

A single, unified evaluation system that works with all trained models
regardless of service or pipeline type. Replaces multiple service-specific
evaluation scripts with one comprehensive solution.

Author: MOrA Team
Version: 3.0
License: MIT
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path

# Data Science Stack
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    mean_absolute_percentage_error, explained_variance_score,
    max_error, median_absolute_error
)
import joblib

# Visualization Stack
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedModelEvaluator:
    """
    Unified model evaluation system that works with any trained model
    Supports both professional and lightweight pipelines
    """
    
    def __init__(self, models_dir: str = "models", data_dir: str = "training_data"):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.evaluation_results = {}
        
        # Industry standards for comparison
        self.industry_standards = {
            'cpu_prediction': {
                'mse_threshold': 0.01,
                'mae_threshold': 0.05,
                'r2_minimum': 0.3,
                'confidence_minimum': 0.7
            },
            'memory_prediction': {
                'mse_threshold': 1e12,
                'mae_threshold': 1e8,
                'r2_minimum': -5.0,
                'confidence_minimum': 0.5
            },
            'replica_prediction': {
                'mse_threshold': 0.5,
                'mae_threshold': 0.3,
                'r2_minimum': 0.4,
                'confidence_minimum': 0.7
            }
        }
    
    def discover_models(self) -> List[str]:
        """Discover all available trained models"""
        logger.info(f"🔍 Discovering models in {self.models_dir}")
        
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory {self.models_dir} does not exist")
            return []
        
        model_files = []
        for file in os.listdir(self.models_dir):
            if file.endswith('.joblib'):
                model_files.append(file)
        
        logger.info(f"📊 Found {len(model_files)} model files")
        return model_files
    
    def load_model(self, model_file: str) -> Optional[Dict[str, Any]]:
        """Load a trained model"""
        model_path = os.path.join(self.models_dir, model_file)
        
        try:
            logger.info(f"📁 Loading model: {model_file}")
            model_data = joblib.load(model_path)
            
            # Extract service name from filename
            service_name = model_file.replace('_lstm_prophet_pipeline.joblib', '').replace('_professional_ml_pipeline.joblib', '')
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"   Service: {service_name}")
            logger.info(f"   Pipeline Type: {model_data.get('pipeline_type', 'unknown')}")
            logger.info(f"   Trained At: {model_data.get('trained_at', 'unknown')}")
            
            return {
                'service_name': service_name,
                'model_data': model_data,
                'model_path': model_path
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_file}: {e}")
            return None
    
    def load_test_data(self, service_name: str) -> Optional[pd.DataFrame]:
        """Load test data for a specific service"""
        logger.info(f"📊 Loading test data for {service_name}")
        
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return None
        
        # Find all CSV files for this service
        csv_files = [f for f in os.listdir(self.data_dir) 
                     if f.startswith(service_name) and f.endswith('.csv')]
        
        if not csv_files:
            logger.warning(f"No test data found for {service_name}")
            return None
        
        logger.info(f"📄 Found {len(csv_files)} data files for {service_name}")
        
        # Load and combine all data
        all_data = []
        for file in csv_files:
            file_path = os.path.join(self.data_dir, file)
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
                logger.info(f"   Loaded {len(df)} rows from {file}")
            except Exception as e:
                logger.error(f"   Failed to load {file}: {e}")
        
        if not all_data:
            logger.error(f"No data could be loaded for {service_name}")
            return None
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Clean timestamp data
        if 'timestamp' in combined_df.columns:
            combined_df['timestamp'] = pd.to_datetime(
                combined_df['timestamp'].astype(str).str.replace('t_', ''), 
                errors='coerce'
            )
            combined_df.dropna(subset=['timestamp'], inplace=True)
            combined_df.sort_values('timestamp', inplace=True)
        
        logger.info(f"📊 Total test data: {len(combined_df)} rows")
        return combined_df
    
    def evaluate_model_performance(self, model_info: Dict[str, Any], test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model performance with comprehensive metrics"""
        service_name = model_info['service_name']
        model_data = model_info['model_data']
        
        logger.info(f"🔍 Evaluating {service_name} model performance...")
        
        evaluation = {
            'service_name': service_name,
            'pipeline_type': model_data.get('pipeline_type', 'unknown'),
            'trained_at': model_data.get('trained_at', 'unknown'),
            'evaluation_timestamp': datetime.now().isoformat(),
            'test_samples': len(test_data),
            'components_status': {},
            'performance_metrics': {},
            'industry_compliance': {},
            'overall_score': 0.0
        }
        
        # Evaluate LSTM models
        lstm_models = model_data.get('lstm_models', {})
        for target, lstm_result in lstm_models.items():
            if lstm_result.get('status') == 'success':
                mse = lstm_result.get('mse', float('inf'))
                mae = lstm_result.get('mae', float('inf'))
                r2 = lstm_result.get('r2', -float('inf'))
                
                evaluation['performance_metrics'][target] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
                
                # Check industry compliance
                compliance = self._check_industry_compliance(target, mse, mae, r2)
                evaluation['industry_compliance'][target] = compliance
        
        # Evaluate fusion results
        fusion_results = model_data.get('fusion_results', {})
        for target, fusion_result in fusion_results.items():
            if fusion_result.get('status') == 'success':
                confidence = fusion_result.get('confidence', 0.0)
                
                if target not in evaluation['performance_metrics']:
                    evaluation['performance_metrics'][target] = {}
                
                evaluation['performance_metrics'][target]['confidence'] = confidence
                
                # Check confidence compliance
                if target in evaluation['industry_compliance']:
                    standards = self.industry_standards.get(f"{target.replace('_target', '_prediction')}", {})
                    evaluation['industry_compliance'][target]['confidence_compliant'] = (
                        confidence >= standards.get('confidence_minimum', 0.5)
                    )
        
        # Calculate overall score
        total_checks = 0
        passed_checks = 0
        
        for target_compliance in evaluation['industry_compliance'].values():
            for check_name, passed in target_compliance.items():
                total_checks += 1
                if passed:
                    passed_checks += 1
        
        if total_checks > 0:
            evaluation['overall_score'] = (passed_checks / total_checks) * 100
        
        return evaluation
    
    def _check_industry_compliance(self, target: str, mse: float, mae: float, r2: float) -> Dict[str, bool]:
        """Check if metrics meet industry standards"""
        if target == 'cpu_target':
            standards = self.industry_standards['cpu_prediction']
        elif target == 'memory_target':
            standards = self.industry_standards['memory_prediction']
        elif target == 'replica_target':
            standards = self.industry_standards['replica_prediction']
        else:
            return {}
        
        return {
            'mse_compliant': mse <= standards['mse_threshold'],
            'mae_compliant': mae <= standards['mae_threshold'],
            'r2_compliant': r2 >= standards['r2_minimum'],
            'overall_compliant': (
                mse <= standards['mse_threshold'] and
                mae <= standards['mae_threshold'] and
                r2 >= standards['r2_minimum']
            )
        }
    
    def generate_evaluation_report(self, evaluation: Dict[str, Any]) -> str:
        """Generate comprehensive evaluation report"""
        service_name = evaluation['service_name']
        overall_score = evaluation['overall_score']
        
        report = []
        report.append("=" * 80)
        report.append(f"🔍 UNIFIED MODEL EVALUATION REPORT - {service_name.upper()}")
        report.append("=" * 80)
        report.append("")
        
        # Model Information
        report.append("📋 MODEL INFORMATION")
        report.append("-" * 40)
        report.append(f"Service: {service_name}")
        report.append(f"Pipeline Type: {evaluation['pipeline_type']}")
        report.append(f"Trained At: {evaluation['trained_at']}")
        report.append(f"Evaluated At: {evaluation['evaluation_timestamp']}")
        report.append(f"Test Samples: {evaluation['test_samples']}")
        report.append("")
        
        # Performance Metrics
        report.append("📊 PERFORMANCE METRICS")
        report.append("-" * 40)
        
        for target, metrics in evaluation['performance_metrics'].items():
            report.append(f"{target}:")
            if 'mse' in metrics:
                report.append(f"  MSE: {metrics['mse']:.6f}")
            if 'mae' in metrics:
                report.append(f"  MAE: {metrics['mae']:.6f}")
            if 'r2' in metrics:
                report.append(f"  R²: {metrics['r2']:.6f}")
            if 'confidence' in metrics:
                report.append(f"  Confidence: {metrics['confidence']:.2f}")
            report.append("")
        
        # Industry Compliance
        report.append("🏭 INDUSTRY COMPLIANCE")
        report.append("-" * 40)
        
        for target, compliance in evaluation['industry_compliance'].items():
            report.append(f"{target}:")
            for check_name, passed in compliance.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                report.append(f"  {check_name}: {status}")
            report.append("")
        
        # Overall Assessment
        report.append("🎯 OVERALL ASSESSMENT")
        report.append("-" * 40)
        report.append(f"Overall Score: {overall_score:.1f}%")
        
        if overall_score >= 80:
            report.append("Status: 🎉 EXCELLENT - Exceeds industry standards")
        elif overall_score >= 60:
            report.append("Status: ✅ GOOD - Meets industry standards")
        elif overall_score >= 40:
            report.append("Status: ⚠️ ACCEPTABLE - Close to industry standards")
        else:
            report.append("Status: ❌ BELOW STANDARDS - Needs improvement")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def evaluate_single_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Evaluate a single service"""
        logger.info(f"🎯 Evaluating single service: {service_name}")
        
        # Find model file
        model_files = self.discover_models()
        model_file = None
        
        for file in model_files:
            if service_name in file:
                model_file = file
                break
        
        if not model_file:
            logger.error(f"❌ No model found for service: {service_name}")
            return None
        
        # Load model
        model_info = self.load_model(model_file)
        if not model_info:
            return None
        
        # Load test data
        test_data = self.load_test_data(service_name)
        if test_data is None:
            logger.warning(f"⚠️ No test data available for {service_name}, using model metrics only")
            test_data = pd.DataFrame()  # Empty dataframe for model-only evaluation
        
        # Evaluate performance
        evaluation = self.evaluate_model_performance(model_info, test_data)
        
        # Generate report
        report = self.generate_evaluation_report(evaluation)
        
        # Save report to organized directory
        os.makedirs("evaluation_reports", exist_ok=True)
        report_path = f"evaluation_reports/{service_name}_evaluation.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Evaluation report saved to: {report_path}")
        
        return {
            'evaluation': evaluation,
            'report': report,
            'report_path': report_path
        }
    
    def evaluate_all_services(self) -> Dict[str, Any]:
        """Evaluate all available services"""
        logger.info("🚀 Starting comprehensive evaluation of all services...")
        
        model_files = self.discover_models()
        if not model_files:
            logger.error("❌ No models found for evaluation")
            return {}
        
        all_evaluations = {}
        summary_stats = {
            'total_services': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'average_score': 0.0,
            'excellent_services': 0,
            'good_services': 0,
            'acceptable_services': 0,
            'below_standard_services': 0
        }
        
        for model_file in model_files:
            try:
                # Extract service name
                service_name = model_file.replace('_lstm_prophet_pipeline.joblib', '').replace('_professional_ml_pipeline.joblib', '')
                
                logger.info(f"🔍 Evaluating {service_name}...")
                
                # Evaluate service
                result = self.evaluate_single_service(service_name)
                
                if result:
                    all_evaluations[service_name] = result['evaluation']
                    summary_stats['successful_evaluations'] += 1
                    
                    score = result['evaluation']['overall_score']
                    if score >= 80:
                        summary_stats['excellent_services'] += 1
                    elif score >= 60:
                        summary_stats['good_services'] += 1
                    elif score >= 40:
                        summary_stats['acceptable_services'] += 1
                    else:
                        summary_stats['below_standard_services'] += 1
                else:
                    summary_stats['failed_evaluations'] += 1
                
                summary_stats['total_services'] += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to evaluate {model_file}: {e}")
                summary_stats['failed_evaluations'] += 1
                summary_stats['total_services'] += 1
        
        # Calculate average score
        if all_evaluations:
            scores = [eval_data['overall_score'] for eval_data in all_evaluations.values()]
            summary_stats['average_score'] = np.mean(scores)
        
        # Generate summary report
        summary_report = self._generate_summary_report(summary_stats, all_evaluations)
        
        # Save summary report to organized directory
        summary_path = f"evaluation_reports/evaluation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary_report)
        
        logger.info(f"📄 Summary report saved to: {summary_path}")
        
        return {
            'evaluations': all_evaluations,
            'summary_stats': summary_stats,
            'summary_report': summary_report,
            'summary_path': summary_path
        }
    
    def _generate_summary_report(self, stats: Dict[str, Any], evaluations: Dict[str, Any]) -> str:
        """Generate summary report for all evaluations"""
        report = []
        report.append("=" * 80)
        report.append("📊 UNIFIED MODEL EVALUATION SUMMARY")
        report.append("=" * 80)
        report.append("")
        
        # Overall Statistics
        report.append("📈 OVERALL STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Services: {stats['total_services']}")
        report.append(f"Successful Evaluations: {stats['successful_evaluations']}")
        report.append(f"Failed Evaluations: {stats['failed_evaluations']}")
        report.append(f"Average Score: {stats['average_score']:.1f}%")
        report.append("")
        
        # Performance Distribution
        report.append("🎯 PERFORMANCE DISTRIBUTION")
        report.append("-" * 40)
        report.append(f"Excellent (≥80%): {stats['excellent_services']}")
        report.append(f"Good (≥60%): {stats['good_services']}")
        report.append(f"Acceptable (≥40%): {stats['acceptable_services']}")
        report.append(f"Below Standard (<40%): {stats['below_standard_services']}")
        report.append("")
        
        # Service-by-Service Results
        report.append("🔍 SERVICE-BY-SERVICE RESULTS")
        report.append("-" * 40)
        
        for service_name, evaluation in evaluations.items():
            score = evaluation['overall_score']
            status = "🎉 EXCELLENT" if score >= 80 else "✅ GOOD" if score >= 60 else "⚠️ ACCEPTABLE" if score >= 40 else "❌ BELOW STANDARDS"
            report.append(f"{service_name}: {score:.1f}% - {status}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Unified Model Evaluation Suite')
    parser.add_argument('--service', type=str, help='Evaluate specific service')
    parser.add_argument('--all', action='store_true', help='Evaluate all services')
    parser.add_argument('--models-dir', type=str, default='models', help='Models directory')
    parser.add_argument('--data-dir', type=str, default='training_data', help='Data directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    evaluator = UnifiedModelEvaluator(
        models_dir=args.models_dir,
        data_dir=args.data_dir
    )
    
    if args.service:
        # Evaluate single service
        result = evaluator.evaluate_single_service(args.service)
        if result:
            print("\n" + result['report'])
        else:
            print(f"❌ Failed to evaluate service: {args.service}")
    
    elif args.all:
        # Evaluate all services
        result = evaluator.evaluate_all_services()
        if result:
            print("\n" + result['summary_report'])
        else:
            print("❌ Failed to evaluate services")
    
    else:
        # Show help
        parser.print_help()

if __name__ == "__main__":
    main()
