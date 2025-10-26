"""
Model library management system for MOrA
"""
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

from ..models.prophet_trainer import ProphetTrainer
from .data_acquisition import DataAcquisitionPipeline
from ..k8s.client import KubernetesClient
from ..monitoring.prometheus_client import PrometheusClient

logger = logging.getLogger(__name__)


class ModelLibrary:
    """
    Manages a library of specialized ML models for different microservices.
    """

    def __init__(
        self,
        model_dir: str = "models",
        namespace: str = "hipster-shop",
        prometheus_url: str = "http://localhost:9090"
    ):
        self.model_dir = model_dir
        self.namespace = namespace
        self.prometheus_url = prometheus_url
        
        # Initialize components
        self.trainer = ProphetTrainer(model_dir=model_dir)
        self.k8s_client = KubernetesClient()
        self.prom_client = PrometheusClient(prometheus_url)
        self.data_pipeline = DataAcquisitionPipeline(
            namespace=namespace,
            prometheus_url=prometheus_url,
            k8s_client=self.k8s_client,
            prom_client=self.prom_client
        )
        
        # Model metadata
        self.metadata_file = os.path.join(model_dir, "model_library.json")
        self.metadata = self._load_metadata()
        
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"ModelLibrary initialized with {len(self.metadata.get('models', {}))} models")

    def _load_metadata(self) -> Dict[str, Any]:
        """Load model library metadata from disk."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load model metadata: {e}")
        
        return {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "models": {}
        }

    def _save_metadata(self):
        """Save model library metadata to disk."""
        try:
            self.metadata["last_updated"] = datetime.now().isoformat()
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model metadata: {e}")

    def train_model_for_service(
        self,
        service_name: str,
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """
        Train Prophet models for all containers in a specific service.
        """
        logger.info(f"Training models for service: {service_name}")
        
        # Check if models already exist
        if not force_retrain and self.has_models_for_service(service_name):
            logger.info(f"Models already exist for {service_name}. Use force_retrain=True to retrain.")
            return {"status": "skipped", "reason": "models_already_exist"}
        
        training_results = {
            "service_name": service_name,
            "start_time": datetime.now().isoformat(),
            "models_trained": [],
            "errors": []
        }
        
        try:
            # Get service containers
            containers = self._get_service_containers(service_name)
            if not containers:
                raise ValueError(f"No containers found for service {service_name}")
            
            logger.info(f"Found containers for {service_name}: {containers}")
            
            # Train models for each container and resource type
            for container_name in containers:
                for resource_type in ["cpu", "memory"]:
                    try:
                        model_key = f"{service_name}_{container_name}_{resource_type}"
                        
                        # Collect training data
                        training_data = self._collect_training_data(
                            service_name, container_name, resource_type
                        )
                        
                        if not training_data or training_data.get("error"):
                            error_msg = f"No training data for {model_key}: {training_data.get('error', 'unknown error')}"
                            logger.warning(error_msg)
                            training_results["errors"].append(error_msg)
                            continue
                        
                        # Train the model
                        logger.info(f"Training {resource_type} model for {container_name}")
                        model_result = self.trainer.train_model(
                            service_name=service_name,
                            container_name=container_name,
                            resource_type=resource_type,
                            metrics_df=training_data["metrics"]
                        )
                        
                        training_results["models_trained"].append(model_result)
                        
                        # Update metadata
                        self.metadata["models"][model_key] = {
                            "service_name": service_name,
                            "container_name": container_name,
                            "resource_type": resource_type,
                            "trained_at": datetime.now().isoformat(),
                            "training_data_points": model_result.get("training_data_points", 0),
                            "evaluation": model_result.get("evaluation", {})
                        }
                        
                        logger.info(f"Successfully trained {model_key}")
                        
                    except Exception as e:
                        error_msg = f"Failed to train {service_name}_{container_name}_{resource_type}: {e}"
                        logger.error(error_msg)
                        training_results["errors"].append(error_msg)
            
            training_results["end_time"] = datetime.now().isoformat()
            training_results["status"] = "completed" if training_results["models_trained"] else "failed"
            
            # Save updated metadata
            self._save_metadata()
            
            logger.info(f"Training completed for {service_name}: {len(training_results['models_trained'])} models trained")
            return training_results
            
        except Exception as e:
            logger.error(f"Failed to train models for {service_name}: {e}")
            training_results["status"] = "failed"
            training_results["errors"].append(str(e))
            return training_results

    def _get_service_containers(self, service_name: str) -> List[str]:
        """Get container names for a service."""
        try:
            deployment = self.k8s_client.get_deployment(service_name, self.namespace)
            if not deployment:
                return []
            
            containers = []
            for container in deployment["containers"]:
                containers.append(container["name"])
            
            return containers
            
        except Exception as e:
            logger.error(f"Failed to get containers for {service_name}: {e}")
            return []

    def _collect_training_data(
        self,
        service_name: str,
        container_name: str,
        resource_type: str,
        duration_hours: int = 48
    ) -> Optional[Dict[str, Any]]:
        """Collect historical training data for a specific service/container/resource."""
        try:
            from datetime import timedelta
            
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=duration_hours)
            
            # Construct Prometheus query
            if resource_type == "cpu":
                query = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{self.namespace}", pod=~"{service_name}-.*", container="{container_name}"}}[1m])) by (container)'
            elif resource_type == "memory":
                query = f'sum(container_memory_working_set_bytes{{namespace="{self.namespace}", pod=~"{service_name}-.*", container="{container_name}"}}) by (container)'
            else:
                return {"error": f"Unsupported resource type: {resource_type}"}
            
            metrics_df = self.prom_client.get_metric_range_data(query, start_time, end_time)
            
            if metrics_df.empty:
                return {"error": "No metrics data available"}
            
            return {
                "service_name": service_name,
                "container_name": container_name,
                "resource_type": resource_type,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "data_points": len(metrics_df),
                "metrics": metrics_df
            }
            
        except Exception as e:
            logger.error(f"Failed to collect training data: {e}")
            return {"error": str(e)}

    def generate_predictions(
        self,
        service_name: str,
        container_name: Optional[str] = None,
        resource_type: Optional[str] = None,
        forecast_hours: int = 168  # 1 week
    ) -> Dict[str, Any]:
        """
        Generate predictions for a service using trained models.
        """
        predictions = {}
        
        try:
            # Get models for the service
            service_models = self.get_service_models(service_name)
            
            if not service_models:
                return {"error": f"No trained models found for service {service_name}"}
            
            # Filter by container and resource type if specified
            filtered_models = []
            for model_key, model_info in service_models.items():
                if container_name and model_info["container_name"] != container_name:
                    continue
                if resource_type and model_info["resource_type"] != resource_type:
                    continue
                filtered_models.append((model_key, model_info))
            
            if not filtered_models:
                return {"error": f"No models match the specified criteria"}
            
            # Generate predictions for each model
            for model_key, model_info in filtered_models:
                try:
                    prediction = self.trainer.predict(
                        service_name=model_info["service_name"],
                        container_name=model_info["container_name"],
                        resource_type=model_info["resource_type"],
                        periods=forecast_hours
                    )
                    
                    predictions[model_key] = prediction
                    
                except Exception as e:
                    logger.error(f"Failed to generate prediction for {model_key}: {e}")
                    predictions[model_key] = {"error": str(e)}
            
            return {
                "service_name": service_name,
                "container_filter": container_name,
                "resource_filter": resource_type,
                "forecast_hours": forecast_hours,
                "generated_at": datetime.now().isoformat(),
                "predictions": predictions
            }
            
        except Exception as e:
            logger.error(f"Failed to generate predictions for {service_name}: {e}")
            return {"error": str(e)}

    def get_service_models(self, service_name: str) -> Dict[str, Any]:
        """Get all trained models for a specific service."""
        service_models = {}
        
        for model_key, model_info in self.metadata.get("models", {}).items():
            if model_info["service_name"] == service_name:
                service_models[model_key] = model_info
        
        return service_models

    def has_models_for_service(self, service_name: str) -> bool:
        """Check if models exist for a specific service."""
        return len(self.get_service_models(service_name)) > 0

    def list_services(self) -> List[str]:
        """List all services that have trained models."""
        services = set()
        
        for model_info in self.metadata.get("models", {}).values():
            services.add(model_info["service_name"])
        
        return list(services)

    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model."""
        return self.metadata.get("models", {}).get(model_key)

    def delete_model(self, model_key: str) -> bool:
        """Delete a specific model from the library."""
        try:
            # Remove from metadata
            if model_key in self.metadata.get("models", {}):
                del self.metadata["models"][model_key]
                self._save_metadata()
            
            # Remove model file
            model_path = os.path.join(self.model_dir, f"{model_key}.pkl")
            if os.path.exists(model_path):
                os.remove(model_path)
            
            # Remove from trainer's memory
            if model_key in self.trainer.trained_models:
                del self.trainer.trained_models[model_key]
            
            logger.info(f"Deleted model {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_key}: {e}")
            return False

    def get_library_status(self) -> Dict[str, Any]:
        """Get overall status of the model library."""
        models = self.metadata.get("models", {})
        
        # Count models by service and resource type
        service_stats = {}
        resource_stats = {"cpu": 0, "memory": 0}
        
        for model_key, model_info in models.items():
            service_name = model_info["service_name"]
            resource_type = model_info["resource_type"]
            
            if service_name not in service_stats:
                service_stats[service_name] = {"cpu": 0, "memory": 0}
            
            service_stats[service_name][resource_type] += 1
            resource_stats[resource_type] += 1
        
        return {
            "total_models": len(models),
            "services_with_models": len(service_stats),
            "service_stats": service_stats,
            "resource_stats": resource_stats,
            "created_at": self.metadata.get("created_at"),
            "last_updated": self.metadata.get("last_updated")
        }

    def bulk_train_services(
        self,
        services: List[str] = None,
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """
        Train models for multiple services in batch.
        """
        if services is None:
            # Default to key services from config
            services = ["frontend", "cartservice", "productcatalogservice", 
                       "checkoutservice", "recommendationservice"]
        
        results = {
            "services_requested": services,
            "training_results": {},
            "start_time": datetime.now().isoformat()
        }
        
        for service_name in services:
            logger.info(f"Training models for {service_name}")
            try:
                service_result = self.train_model_for_service(service_name, force_retrain)
                results["training_results"][service_name] = service_result
            except Exception as e:
                logger.error(f"Failed to train {service_name}: {e}")
                results["training_results"][service_name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        results["end_time"] = datetime.now().isoformat()
        results["summary"] = {
            "total_services": len(services),
            "successful": len([r for r in results["training_results"].values() 
                             if r.get("status") == "completed"]),
            "failed": len([r for r in results["training_results"].values() 
                          if r.get("status") == "failed"])
        }
        
        return results
