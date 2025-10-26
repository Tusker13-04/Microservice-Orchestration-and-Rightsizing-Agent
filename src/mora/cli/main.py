"""
Main CLI entry point for MOrA
"""
import click
import yaml
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from datetime import datetime

try:
    from ..core.data_pipeline import DataPipeline
    from ..core.statistical_strategy import StatisticalRightsizer
    from ..core.model_library import ModelLibrary
    from ..core.data_acquisition import DataAcquisitionPipeline
    from ..utils.config import load_config
except ImportError:
    # Handle direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.mora.core.data_pipeline import DataPipeline
    from src.mora.core.statistical_strategy import StatisticalRightsizer
    from src.mora.core.model_library import ModelLibrary
    from src.mora.core.data_acquisition import DataAcquisitionPipeline
    from src.mora.utils.config import load_config

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """MOrA - Microservices-Aware Orchestrator Agent for Predictive Kubernetes Rightsizing"""
    pass


@main.command()
@click.option('--strategy', type=click.Choice(['statistical', 'predictive']), 
              default='statistical', help='Rightsizing strategy to use')
@click.option('--service', required=True, help='Target microservice name')
@click.option('--namespace', default='hipster-shop', help='Kubernetes namespace')
@click.option('--prometheus-url', default='http://localhost:9090', help='Prometheus URL')
@click.option('--duration-hours', default=24, help='Hours of historical data to analyze')
@click.option('--output-format', type=click.Choice(['table', 'yaml', 'json']), 
              default='table', help='Output format')
def rightsize(strategy, service, namespace, prometheus_url, duration_hours, output_format):
    """
    Generate rightsizing recommendations for a microservice
    """
    console.print(f"[bold blue]MOrA Rightsizing Analysis[/bold blue]")
    console.print(f"Strategy: {strategy}")
    console.print(f"Service: {service}")
    console.print(f"Namespace: {namespace}")
    console.print(f"Analysis Period: {duration_hours} hours")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing analysis...", total=None)

            pipeline = DataPipeline(namespace=namespace, prometheus_url=prometheus_url)

            progress.update(task, description="Checking connections...")
            connections = pipeline.test_connections()

            if not connections['kubernetes'] or not connections['prometheus']:
                console.print("\n[red]Cannot connect to required services:[/red]")
                for service_name, status in connections.items():
                    status_icon = "❌" if not status else "✅"
                    console.print(f"  {status_icon} {service_name.capitalize()}")
                return

            if strategy == 'statistical':
                progress.update(task, description=f"Collecting data for {service}...")
                service_data = pipeline.collect_service_data(service, duration_hours * 60)

                progress.update(task, description="Generating statistical recommendations...")
                rightsizer = StatisticalRightsizer(cpu_percentile=95.0, memory_buffer_percentage=15.0)
                recommendations = rightsizer.generate_recommendations(service_data)

                if not recommendations:
                    console.print("\n[red]No recommendations generated. Check data availability.[/red]")
                    return

                validation_results = rightsizer.validate_recommendations(recommendations)

                if output_format == 'json':
                    console.print(json.dumps(recommendations, indent=2, default=str))
                elif output_format == 'yaml':
                    console.print(yaml.dump(recommendations, default_flow_style=False, default_representer=yaml.dumper.SafeDumper))
                else:
                    table = Table(title=f"Rightsizing Recommendations - {service}")
                    table.add_column("Container", style="cyan")
                    table.add_column("Resource", style="yellow")
                    table.add_column("Current Request", style="magenta")
                    table.add_column("Recommended Request", style="green")
                    table.add_column("Analysis", style="dim")

                    for rec in recommendations:
                        container_name = rec['container_name']
                        current = rec['current_requests']
                        recommended = rec['recommended_requests']

                        cpu_analysis = rec['analysis']['cpu']
                        cpu_analysis_text = f"P{rightsizer.cpu_percentile:.0f}: {cpu_analysis.get('percentile_value', 0):.3f} cores" if cpu_analysis.get('has_data') else "No data"

                        table.add_row(
                            container_name,
                            "CPU",
                            str(current['cpu']),
                            recommended['cpu'],
                            cpu_analysis_text
                        )

                        memory_analysis = rec['analysis']['memory']
                        memory_analysis_text = f"Max: {memory_analysis.get('max_usage_bytes', 0) / (1024*1024):.0f} MiB" if memory_analysis.get('has_data') else "No data"

                        table.add_row(
                            "",
                            "Memory",
                            str(current['memory']),
                            recommended['memory'],
                            memory_analysis_text
                        )

                    console.print(table)

            elif strategy == 'predictive':
                progress.update(task, description="Initializing model library...")
                model_library = ModelLibrary(namespace=namespace, prometheus_url=prometheus_url)

                progress.update(task, description=f"Generating predictive recommendations for {service}...")
                
                # Generate predictions using trained models
                predictions = model_library.generate_predictions(
                    service_name=service,
                    forecast_hours=24 * 7  # 1 week
                )

                if predictions.get('error'):
                    console.print(f"\n[yellow]Predictive analysis not available: {predictions['error']}[/yellow]")
                    console.print("Run 'mora train --service {service}' to train models first")
                else:
                    # Process predictions into recommendations
                    console.print(f"\n[green]Predictive recommendations for {service}:[/green]")
                    
                    if output_format == 'json':
                        console.print(json.dumps(predictions, indent=2, default=str))
                    elif output_format == 'yaml':
                        console.print(yaml.dump(predictions, default_flow_style=False, default_representer=yaml.dumper.SafeDumper))
                    else:
                        # Display predictions in table format
                        for model_key, model_prediction in predictions.get('predictions', {}).items():
                            if not model_prediction.get('error'):
                                console.print(f"\n[bold]{model_key}:[/bold]")
                                for pred in model_prediction.get('predictions', []):
                                    console.print(f"  {pred['timestamp']}: {pred['predicted_value']:.3f}")

            if strategy == 'statistical' and validation_results.get('warnings'):
                console.print("\n[yellow]Recommendation Validation Warnings:[/yellow]")
                for warning in validation_results['warnings']:
                    console.print(f"  ⚠️  {warning}")

    except Exception as e:
        console.print(f"\n[red]Error during rightsizing analysis: {e}[/red]")
        console.print("[yellow]Ensure the service exists and monitoring is properly configured[/yellow]")


@main.command()
@click.option('--namespace', default='hipster-shop', help='Kubernetes namespace')
@click.option('--prometheus-url', default='http://localhost:9090', help='Prometheus URL')
def status(namespace, prometheus_url):
    """
    Show current status of monitoring stack and services
    """
    console.print(f"[bold blue]MOrA System Status[/bold blue]")
    console.print(f"Namespace: {namespace}")
    console.print(f"Prometheus URL: {prometheus_url}")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Connecting to services...", total=None)

            # Load Grafana URL from config
            try:
                config = load_config('config/default.yaml')
                grafana_url = config.get('grafana', {}).get('url', 'http://localhost:4000')
            except:
                grafana_url = 'http://localhost:4000'
            
            pipeline = DataPipeline(namespace=namespace, prometheus_url=prometheus_url, grafana_url=grafana_url)

            progress.update(task, description="Testing connections...")
            connections = pipeline.test_connections()

            progress.update(task, description="Gathering system information...")
            summary = pipeline.get_system_summary()

        console.print("\n[bold]Connection Status:[/bold]")
        for service, status in connections.items():
            status_icon = "✅" if status else "❌"
            console.print(f"  {status_icon} {service.capitalize()}: {'Connected' if status else 'Not available'}")

        if 'error' not in summary:
            console.print(f"\n[bold]System Summary:[/bold]")
            console.print(f"  Namespace: {summary['namespace']}")
            console.print(f"  Total Services: {summary['total_services']}")

            if summary['service_stats']:
                console.print(f"\n[bold]Service Status:[/bold]")
                service_table = Table()
                service_table.add_column("Service", style="cyan")
                service_table.add_column("Replicas", style="magenta")
                service_table.add_column("Ready", style="green")
                service_table.add_column("Containers", style="yellow")

                for service_name, stats in summary['service_stats'].items():
                    service_table.add_row(
                        service_name,
                        str(stats['replicas']),
                        str(stats['ready_replicas']),
                        str(stats['containers'])
                    )

                console.print(service_table)
        else:
            console.print(f"\n[red]Error getting system summary: {summary['error']}[/red]")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        console.print("[yellow]Make sure Minikube is running and services are deployed[/yellow]")


@main.command(name='setup-grafana')
@click.option('--namespace', default='hipster-shop', help='Kubernetes namespace')
@click.option('--prometheus-url', default='http://localhost:9090', help='Prometheus URL')
@click.option('--grafana-url', default='http://localhost:4000', help='Grafana URL')
def setup_grafana(namespace, prometheus_url, grafana_url):
    """
    Set up Grafana dashboard integration for MOrA monitoring
    """
    console.print(f"[bold blue]MOrA Grafana Integration Setup[/bold blue]")
    console.print(f"Namespace: {namespace}")
    console.print(f"Grafana URL: {grafana_url}")
    console.print(f"Prometheus URL: {prometheus_url}")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing data pipeline...", total=None)

            pipeline = DataPipeline(
                namespace=namespace, 
                prometheus_url=prometheus_url, 
                grafana_url=grafana_url
            )

            progress.update(task, description="Testing Grafana connection...")
            grafana_status = pipeline.grafana_client.test_connection()

            if not grafana_status:
                console.print(f"\n[red]❌ Cannot connect to Grafana at {grafana_url}[/red]")
                console.print("[yellow]Make sure Grafana is running and port-forward is active[/yellow]")
                return

            progress.update(task, description="Setting up MOrA dashboard...")
            setup_result = pipeline.setup_grafana_integration()

        if setup_result['success']:
            console.print(f"\n[green]✅ Grafana integration setup completed![/green]")
            console.print(f"Dashboard UID: {setup_result['dashboard_uid']}")
            console.print(f"Dashboard URL: {setup_result['dashboard_url']}")
            console.print(f"\n[bold]What's been set up:[/bold]")
            console.print("  - MOrA monitoring dashboard with CPU, Memory, Network, and Replica panels")
            console.print("  - Prometheus data source verification")
            console.print("  - Real-time metrics visualization for your microservices")
        else:
            console.print(f"\n[red]❌ Grafana setup failed[/red]")
            console.print(f"Error: {setup_result['error']}")

    except Exception as e:
        console.print(f"\n[red]Error during Grafana setup: {e}[/red]")


@main.group()
def train():
    """Model training commands for Phase 2 functionality"""
    pass


@train.command()
@click.option('--service', help='Specific service to train (if not provided, trains all services)')
@click.option('--namespace', default='hipster-shop', help='Kubernetes namespace')
@click.option('--prometheus-url', default='http://localhost:9090', help='Prometheus URL')
@click.option('--force-retrain', is_flag=True, help='Force retrain even if models exist')
def models(service, namespace, prometheus_url, force_retrain):
    """
    Train ML models for predictive rightsizing
    """
    console.print(f"[bold blue]MOrA Model Training[/bold blue]")
    console.print(f"Target: {service if service else 'All services'}")
    console.print(f"Namespace: {namespace}")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing model library...", total=None)

            model_library = ModelLibrary(
                namespace=namespace,
                prometheus_url=prometheus_url
            )

            progress.update(task, description="Training models...")
            
            if service:
                # Train specific service
                result = model_library.train_model_for_service(service, force_retrain)
                
                if result['status'] == 'completed':
                    console.print(f"\n[green]✅ Training completed for {service}[/green]")
                    console.print(f"Models trained: {len(result['models_trained'])}")
                    if result['errors']:
                        console.print(f"Errors: {len(result['errors'])}")
                else:
                    console.print(f"\n[red]❌ Training failed for {service}[/red]")
                    for error in result.get('errors', []):
                        console.print(f"  Error: {error}")
            else:
                # Train all services
                result = model_library.bulk_train_services(force_retrain=force_retrain)
                
                console.print(f"\n[bold]Training Results:[/bold]")
                console.print(f"Services requested: {result['summary']['total_services']}")
                console.print(f"Successful: {result['summary']['successful']}")
                console.print(f"Failed: {result['summary']['failed']}")

                # Show detailed results
                for service_name, service_result in result['training_results'].items():
                    status_icon = "✅" if service_result.get('status') == 'completed' else "❌"
                    console.print(f"  {status_icon} {service_name}")

    except Exception as e:
        console.print(f"\n[red]Error during model training: {e}[/red]")


@train.command()
@click.option('--service', required=True, help='Service to run clean training experiments')
@click.option('--config-file', default='config/default.yaml', help='Configuration file path')
def clean_experiments(service, config_file):
    """
    Run clean steady-state training experiments (Phase 2).
    Each experiment: fixed replicas + fixed load = clean data.
    """
    console.print(f"[bold blue]MOrA Clean Training Experiments[/bold blue]")
    console.print(f"Target Service: {service}")
    console.print(f"Config File: {config_file}")
    
    try:
        # Load training configuration from config file
        config = load_config(config_file)
        training_config = config.get('training', {}).get('steady_state_config', {})
        
        console.print(f"\n[bold]Training Configuration:[/bold]")
        console.print(f"  Experiment Duration: {training_config.get('experiment_duration_minutes', 45)} minutes")
        console.print(f"  Replica Counts: {training_config.get('replica_counts', [1, 2, 4, 6])}")
        console.print(f"  Load Levels: {training_config.get('load_levels_users', [10, 50, 100, 150, 200, 250])} users")
        console.print(f"  Test Scenarios: {training_config.get('test_scenarios', ['browsing'])}")
        console.print(f"  Sample Interval: {training_config.get('sample_interval', '15s')}")
        
        # Updated calculation to include scenarios (triple-loop)
        replica_counts = len(training_config.get('replica_counts', [1, 2, 4, 6]))
        load_levels = len(training_config.get('load_levels_users', [10, 50, 100, 150, 200, 250]))
        scenarios = len(training_config.get('test_scenarios', ['browsing']))
        total_experiments = replica_counts * load_levels * scenarios
        console.print(f"  Total Experiments: {total_experiments} (= {scenarios} scenarios × {replica_counts} replicas × {load_levels} load levels)")
    
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load config file, using defaults: {e}[/yellow]")
        training_config = {}

    try:
        # Load namespace and prometheus URL from config
        if not training_config:
            # Load defaults if config file wasn't loaded properly
            config = load_config(config_file)
        else:
            config = load_config(config_file) if 'config' not in locals() else config
            
        namespace = config.get('kubernetes', {}).get('namespace', 'hipster-shop')
        prometheus_url = config.get('prometheus', {}).get('url', 'http://localhost:9090')

        # Check existing progress before starting
        try:
            temp_pipeline = DataAcquisitionPipeline(
                namespace=namespace,
                prometheus_url=prometheus_url
            )
            completed_experiments = temp_pipeline._get_completed_experiments(service)
            
            console.print(f"\n[bold]Progress Status:[/bold]")
            console.print(f"  Completed: {len(completed_experiments)}")
            console.print(f"  Remaining: {total_experiments - len(completed_experiments)}")
            
            if len(completed_experiments) > 0:
                console.print(f"\n[yellow]🔄 Resuming from where you left off![/yellow]")
                console.print(f"Found {len(completed_experiments)} completed experiments that will be skipped.")
            else:
                console.print(f"\n[blue]🚀 Starting fresh training session[/blue]")
            
        except Exception as e:
            console.print(f"[yellow]Could not check existing progress: {e}[/yellow]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing data acquisition pipeline...", total=None)

            data_pipeline = DataAcquisitionPipeline(
                namespace=namespace,
                prometheus_url=prometheus_url
            )

            progress.update(task, description="Running clean training experiments...")
            
            # Run the clean training experiments with the loaded config
            result = data_pipeline.run_isolated_training_experiment(
                target_service=service,
                config=training_config
            )

            if result['status'] in ['completed', 'completed_with_warnings']:
                console.print(f"\n[green]✅ Clean training completed for {service}[/green]")
                console.print(f"Total experiments: {result.get('total_combinations', 0)}")
                console.print(f"Experiments completed: {len(result.get('experiments', []))}")
                
                # Show summary of results
                experiments = result.get('experiments', [])
                successful = len([e for e in experiments if e.get('status') == 'completed'])
                with_warnings = len([e for e in experiments if e.get('status') == 'completed_with_warnings'])
                failed = len([e for e in experiments if e.get('status') == 'failed'])
                
                console.print(f"Successful: {successful}")
                if with_warnings > 0:
                    console.print(f"Completed with warnings: {with_warnings}")
                console.print(f"Failed: {failed}")
                console.print(f"\n[bold]Each experiment:[/bold] {training_config.get('experiment_duration_minutes', 45)} minutes of steady-state data")
                
                # Show data quality summary if available
                quality_stats = {"passed": 0, "warnings": 0, "failed": 0}
                for exp in experiments:
                    if 'data_quality' in exp:
                        quality_status = exp['data_quality'].get('status', 'unknown')
                        if quality_status == 'passed':
                            quality_stats['passed'] += 1
                        elif quality_status == 'warnings':
                            quality_stats['warnings'] += 1
                        else:
                            quality_stats['failed'] += 1
                
                if sum(quality_stats.values()) > 0:
                    console.print(f"\n[bold]Data Quality Summary:[/bold]")
                    console.print(f"  Passed: {quality_stats['passed']}")
                    console.print(f"  Warnings: {quality_stats['warnings']}")
                    console.print(f"  Failed: {quality_stats['failed']}")
                
            else:
                console.print(f"\n[red]❌ Training failed for {service}[/red]")
                if result.get('error'):
                    console.print(f"Error: {result['error']}")

    except Exception as e:
        console.print(f"\n[red]Error during clean training experiments: {e}[/red]")
        console.print("[yellow]Make sure Minikube is running and the configuration file is valid[/yellow]")


@train.command()
@click.option('--service', required=True, help='Service to check training progress')
@click.option('--config-file', default='config/default.yaml', help='Configuration file path')
def status(service, config_file):
    """
    Check training experiment progress for a service.
    """
    console.print(f"[bold blue]Training Progress for {service}[/bold blue]")
    
    try:
        # Load configuration
        config = load_config(config_file)
        namespace = config.get('kubernetes', {}).get('namespace', 'hipster-shop')
        prometheus_url = config.get('prometheus', {}).get('url', 'http://localhost:9090')
        
        # Get training config for experiment count
        training_config = config.get('training', {}).get('steady_state_config', {})
        replica_counts = len(training_config.get('replica_counts', [1, 2, 4, 6]))
        load_levels = len(training_config.get('load_levels_users', [10, 50, 100, 150, 200, 250]))
        scenarios = len(training_config.get('test_scenarios', ['browsing']))
        total_experiments = replica_counts * load_levels * scenarios
        
        # Initialize pipeline and check progress
        pipeline = DataAcquisitionPipeline(
            namespace=namespace,
            prometheus_url=prometheus_url
        )
        completed_experiments = pipeline._get_completed_experiments(service)
        
        console.print(f"\n[bold]Progress Summary:[/bold]")
        console.print(f"  Total Experiments: {total_experiments}")
        console.print(f"  Completed: {len(completed_experiments)}")
        console.print(f"  Remaining: {total_experiments - len(completed_experiments)}")
        
        if len(completed_experiments) > 0:
            console.print(f"\n[green]✅ Completed Experiments:[/green]")
            # Show first 10 completed experiments
            for exp_id in list(completed_experiments)[:10]:
                console.print(f"  • {exp_id}")
            if len(completed_experiments) > 10:
                console.print(f"  ... and {len(completed_experiments) - 10} more")
        else:
            console.print(f"\n[blue]ℹ️  No experiments completed yet[/blue]")
            
        if len(completed_experiments) < total_experiments:
            console.print(f"\n[yellow]💡 Run 'mora train clean-experiments --service {service}' to continue/resume training[/yellow]")
        else:
            console.print(f"\n[green]🎉 All experiments completed! Ready for model training.[/green]")
            
    except Exception as e:
        console.print(f"[red]Error checking status: {e}[/red]")


@train.command()
@click.option('--services', required=True, help='Comma-separated list of services to train in parallel')
@click.option('--config-file', default='config/resource-optimized.yaml', help='Configuration file path')
@click.option('--max-workers', default=1, help='Maximum number of parallel workers')
def parallel_experiments(services, config_file, max_workers):
    """
    Run clean steady-state training experiments in parallel across multiple services.
    Dramatically reduces total collection time.
    """
    service_list = [s.strip() for s in services.split(',')]
    
    console.print(f"[bold blue]MOrA Parallel Training Experiments[/bold blue]")
    console.print(f"Target Services: {service_list}")
    console.print(f"Max Workers: {max_workers}")
    console.print(f"Config File: {config_file}")
    
    try:
        # Load training configuration from config file
        config = load_config(config_file)
        training_config = config.get('training', {}).get('steady_state_config', {})
        
        console.print(f"\n[bold]Training Configuration:[/bold]")
        console.print(f"  Experiment Duration: {training_config.get('experiment_duration_minutes', 45)} minutes")
        console.print(f"  Replica Counts: {training_config.get('replica_counts', [1, 2, 4, 6])}")
        console.print(f"  Load Levels: {training_config.get('load_levels_users', [10, 50, 100, 150, 200, 250])} users")
        console.print(f"  Test Scenarios: {training_config.get('test_scenarios', ['browsing', 'checkout'])}")
        
        # Calculate total experiments
        replica_counts = len(training_config.get('replica_counts', [1, 2, 4, 6]))
        load_levels = len(training_config.get('load_levels_users', [10, 50, 100, 150, 200, 250]))
        scenarios = len(training_config.get('test_scenarios', ['browsing', 'checkout']))
        total_experiments = len(service_list) * replica_counts * load_levels * scenarios
        
        console.print(f"\n[bold]Parallel Execution Plan:[/bold]")
        console.print(f"  Total Experiments: {total_experiments} (= {len(service_list)} services × {scenarios} scenarios × {replica_counts} replicas × {load_levels} load levels)")
        
        # Estimate time savings
        sequential_time_hours = total_experiments * training_config.get('experiment_duration_minutes', 45) / 60
        parallel_time_hours = sequential_time_hours / min(max_workers, len(service_list))
        
        console.print(f"  Sequential Time: ~{sequential_time_hours:.1f} hours")
        console.print(f"  Parallel Time: ~{parallel_time_hours:.1f} hours (estimated {sequential_time_hours/parallel_time_hours:.1f}x speedup)")
        
        # Get namespace and prometheus URL
        namespace = config.get('kubernetes', {}).get('namespace', 'hipster-shop')
        prometheus_url = config.get('prometheus', {}).get('url', 'http://localhost:9090')
        
        # Initialize pipeline
        pipeline = DataAcquisitionPipeline(
            namespace=namespace,
            prometheus_url=prometheus_url
        )
        
        console.print(f"\n🚀 Starting parallel training for {len(service_list)} services...")
        
        # Run parallel experiments
        results = pipeline.run_parallel_training_experiments(service_list, config, max_workers=max_workers)
        
        if results.get('status') == 'completed':
            console.print(f"\n[green]✅ Parallel training completed![/green]")
            console.print(f"Total experiments: {len(results['experiments'])}")
            
            # Count different statuses
            successful = len([e for e in results['experiments'] if e.get('status') == 'completed'])
            failed = len([e for e in results['experiments'] if e.get('status') == 'failed'])
            skipped = len([e for e in results['experiments'] if e.get('status') == 'skipped'])
            
            console.print(f"Successful: {successful}")
            if skipped > 0:
                console.print(f"Skipped (already completed): {skipped}")
            if failed > 0:
                console.print(f"Failed: {failed}")
                console.print("[yellow]Check logs for failed experiment details[/yellow]")
            
            # Show data quality summary
            quality_passed = len([e for e in results['experiments'] 
                                if e.get('status') == 'completed' and 
                                e.get('data_quality', {}).get('status') == 'passed'])
            if quality_passed > 0:
                console.print(f"\n[green]Data Quality: {quality_passed}/{successful} experiments passed quality checks[/green]")
        else:
            console.print(f"\n[red]❌ Parallel training failed: {results.get('error', 'Unknown error')}[/red]")
        
    except Exception as e:
        console.print(f"\n[red]Error during parallel training experiments: {e}[/red]")
        console.print("[yellow]Make sure Minikube is running and the configuration file is valid[/yellow]")


@train.command()
@click.option('--service', required=True, help='Service for dynamic evaluation experiment')
@click.option('--config-file', default='config/default.yaml', help='Configuration file path')
def dynamic_evaluation(service, config_file):
    """
    Run dynamic evaluation experiment (Phase 4).
    Changes load every 15-30 minutes to test system response.
    """
    console.print(f"[bold yellow]MOrA Dynamic Evaluation (Phase 4)[/bold yellow]")
    console.print(f"Target Service: {service}")
    console.print(f"Config File: {config_file}")
    
    try:
        # Load evaluation configuration from config file
        config = load_config(config_file)
        evaluation_config = config.get('evaluation', {})
        
        console.print(f"\n[bold]Evaluation Configuration:[/bold]")
        console.print(f"  Total Duration: {evaluation_config.get('collection_duration_minutes', 120)} minutes")
        console.print(f"  Sample Interval: {evaluation_config.get('sample_interval', '30s')}")
        
        scenarios = evaluation_config.get('dynamic_load_scenarios', [])
        console.print(f"  Dynamic Scenarios: {len(scenarios)}")
        for scenario in scenarios:
            console.print(f"    - {scenario.get('name', 'unknown')}: {scenario.get('users', 0)} users for {scenario.get('duration_minutes', 0)} min")
        
        console.print("\n[yellow]Note: This should be run after models are trained![/yellow]")
        
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load config file, using defaults: {e}[/yellow]")
        evaluation_config = {}

    try:
        # Load namespace and prometheus URL from config
        config = load_config(config_file)
        namespace = config.get('kubernetes', {}).get('namespace', 'hipster-shop')
        prometheus_url = config.get('prometheus', {}).get('url', 'http://localhost:9090')

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing data acquisition pipeline...", total=None)

            data_pipeline = DataAcquisitionPipeline(
                namespace=namespace,
                prometheus_url=prometheus_url
            )

            progress.update(task, description="Running dynamic evaluation experiment...")
            
            # Run the dynamic evaluation experiment
            result = data_pipeline.run_dynamic_evaluation_experiment(
                target_service=service,
                config=evaluation_config
            )

            if result['status'] == 'completed':
                console.print(f"\n[green]✅ Dynamic evaluation completed for {service}[/green]")
                console.print(f"Scenarios completed: {len(result.get('scenarios', []))}")
                
                # Show summary of scenarios
                scenarios = result.get('scenarios', [])
                successful = len([s for s in scenarios if s.get('status') == 'completed'])
                failed = len([s for s in scenarios if s.get('status') == 'failed'])
                
                console.print(f"Successful scenarios: {successful}")
                console.print(f"Failed scenarios: {failed}")
                
            else:
                console.print(f"\n[red]❌ Dynamic evaluation failed for {service}[/red]")
                if result.get('error'):
                    console.print(f"Error: {result['error']}")

    except Exception as e:
        console.print(f"\n[red]Error during dynamic evaluation: {e}[/red]")
        console.print("[yellow]Make sure Minikube is running and models are trained[/yellow]")


@train.command()
@click.option('--services', help='Comma-separated list of services to process (if not provided, processes all)')
@click.option('--config-file', default='config/default.yaml', help='Configuration file path')
@click.option('--output-file', help='Output CSV file path (default: data/training_data_master.csv)')
def process_data(services, config_file, output_file):
    """
    Process collected experiment data into training dataset.
    Converts JSON experiment files and CSV metrics into unified DataFrame.
    """
    service_list = [s.strip() for s in services.split(',')] if services else None
    
    console.print(f"[bold blue]MOrA Data Processing[/bold blue]")
    console.print(f"Target Services: {service_list or 'All services'}")
    console.print(f"Config File: {config_file}")
    
    try:
        # Load configuration
        config = load_config(config_file)
        namespace = config.get('kubernetes', {}).get('namespace', 'hipster-shop')
        prometheus_url = config.get('prometheus', {}).get('url', 'http://localhost:9090')
        
        # Initialize data pipeline
        data_pipeline = DataAcquisitionPipeline(
            namespace=namespace,
            prometheus_url=prometheus_url
        )
        
        console.print(f"\n🔄 Processing collected experiment data...")
        
        # Process the data
        result = data_pipeline.process_collected_data_for_training(
            target_services=service_list,
            output_file=output_file
        )
        
        if result["status"] == "completed":
            console.print(f"\n[green]✅ Data processing completed![/green]")
            console.print(f"Output file: {result['output_file']}")
            console.print(f"Experiments processed: {result['experiments_processed']}")
            console.print(f"Total rows: {result['total_rows']}")
            console.print(f"Data shape: {result['shape']}")
            
            # Show some column info
            columns = result['columns']
            console.print(f"\nColumns: {len(columns)} total")
            console.print("Key columns: " + ", ".join([col for col in columns if col in ['service', 'scenario', 'replica_count', 'load_users', 'requests_per_second']]))
            
        elif result["status"] == "no_data":
            console.print(f"\n[yellow]⚠️  No experiment data found[/yellow]")
            console.print("Run data collection experiments first with: train parallel-experiments")
            
        else:
            console.print(f"\n[red]❌ Data processing failed[/red]")
            console.print(f"Error: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        console.print(f"\n[red]Error during data processing: {e}[/red]")


@train.command()
@click.option('--services', required=True, help='Comma-separated list of services to train models for')
@click.option('--config-file', default='config/default.yaml', help='Configuration file path')
@click.option('--training-data-file', help='Path to training data CSV (default: data/training_data_master.csv)')
def train_models(services, config_file, training_data_file):
    """
    Train Prophet models using collected experiment data.
    """
    service_list = [s.strip() for s in services.split(',')]
    
    console.print(f"[bold blue]MOrA Model Training[/bold blue]")
    console.print(f"Target Services: {service_list}")
    console.print(f"Config File: {config_file}")
    
    try:
        # Load configuration
        config = load_config(config_file)
        namespace = config.get('kubernetes', {}).get('namespace', 'hipster-shop')
        prometheus_url = config.get('prometheus', {}).get('url', 'http://localhost:9090')
        
        # Initialize model library
        from ..core.model_library import ModelLibrary
        model_library = ModelLibrary(
            namespace=namespace,
            prometheus_url=prometheus_url
        )
        
        console.print(f"\n🔄 Training models for {len(service_list)} services...")
        
        total_trained = 0
        total_errors = 0
        
        for service in service_list:
            console.print(f"\n📊 Training models for {service}...")
            
            try:
                result = model_library.train_service_models(service)
                
                if result["status"] == "completed":
                    trained_count = len(result["models_trained"])
                    total_trained += trained_count
                    console.print(f"  ✅ Trained {trained_count} models for {service}")
                else:
                    total_errors += 1
                    console.print(f"  ❌ Failed to train models for {service}")
                    for error in result.get("errors", []):
                        console.print(f"    Error: {error}")
                        
            except Exception as e:
                total_errors += 1
                console.print(f"  ❌ Error training {service}: {e}")
        
        console.print(f"\n[green]🎉 Model training completed![/green]")
        console.print(f"Services trained: {len(service_list) - total_errors}/{len(service_list)}")
        console.print(f"Total models trained: {total_trained}")
        
    except Exception as e:
        console.print(f"\n[red]Error during model training: {e}[/red]")


@main.command()
@click.option('--namespace', default='hipster-shop', help='Kubernetes namespace')
@click.option('--prometheus-url', default='http://localhost:9090', help='Prometheus URL')
def models_status(namespace, prometheus_url):
    """
    Show status of trained models
    """
    console.print(f"[bold blue]MOrA Model Library Status[/bold blue]")

    try:
        model_library = ModelLibrary(
            namespace=namespace,
            prometheus_url=prometheus_url
        )
        
        status = model_library.get_library_status()
        
        console.print(f"\n[bold]Library Overview:[/bold]")
        console.print(f"Total Models: {status['total_models']}")
        console.print(f"Services with Models: {status['services_with_models']}")
        console.print(f"Created: {status['created_at']}")
        console.print(f"Last Updated: {status['last_updated']}")
        
        if status['service_stats']:
            console.print(f"\n[bold]Services with Models:[/bold]")
            for service_name, stats in status['service_stats'].items():
                console.print(f"  {service_name}: {stats['cpu']} CPU models, {stats['memory']} Memory models")
        
        # List all services
        services = model_library.list_services()
        if services:
            console.print(f"\n[bold]Available Services:[/bold]")
            for service_name in services:
                models = model_library.get_service_models(service_name)
                console.print(f"  {service_name}: {len(models)} models")

    except Exception as e:
        console.print(f"\n[red]Error getting model status: {e}[/red]")


if __name__ == '__main__':
    main()
