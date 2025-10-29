# MOrA DevOps Infrastructure

This document describes the DevOps infrastructure and deployment strategies for the MOrA (Microservices-Aware Orchestrator Agent) project.

## 🚀 Quick Start

### Local Development with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Access Points
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **MOrA CLI**: Available in `mora-cli` container

## 🏗️ Infrastructure Components

### 1. GitHub Actions CI/CD Pipeline

**Location**: `.github/workflows/ci.yml`

**Features**:
- Multi-Python version testing (3.8, 3.9, 3.10, 3.11)
- Automated linting with flake8 and black
- Test coverage reporting with Codecov integration
- Docker image building and pushing to Docker Hub
- Security scanning with Trivy
- Matrix strategy for comprehensive testing

**Triggers**:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

### 2. Docker Containerization

**Dockerfile Features**:
- Multi-stage build for optimized image size
- Python 3.11 slim base image
- Build dependencies isolated in builder stage
- Production-ready runtime environment
- Proper layer caching for faster builds

**Image Size**: ~500MB (optimized with multi-stage build)

### 3. Docker Compose Development Environment

**Services**:
- `mora-cli`: Main application container
- `mora-prometheus`: Metrics collection and storage
- `mora-grafana`: Visualization and dashboards

**Features**:
- Volume mounting for persistent data
- Network isolation with custom bridge network
- Environment variable configuration
- Health checks and restart policies

### 4. Kubernetes Deployment

**Location**: `k8s/deployment.yaml`

**Components**:
- Deployment with 1 replica
- Service for internal communication
- PersistentVolumeClaims for data persistence
- Resource limits and requests
- Environment variable configuration

## 📊 Monitoring Stack

### Prometheus Configuration
- **File**: `config/prometheus.yml`
- **Scrape Targets**:
  - Prometheus self-monitoring
  - MOrA CLI application metrics
  - Node exporter (when available)

### Grafana Configuration
- **File**: `config/grafana/dashboards/dashboard.yml`
- **Default Credentials**: admin/admin
- **Dashboard Provisioning**: Automatic dashboard loading

## 🔧 Development Workflow

### 1. Local Development
```bash
# Start development environment
docker-compose up -d

# Run tests
docker-compose exec mora-cli python -m pytest tests/

# Access application
docker-compose exec mora-cli python -m src.mora.cli.main --help
```

### 2. CI/CD Pipeline
1. **Code Push**: Triggers automated pipeline
2. **Testing**: Multi-version Python testing
3. **Linting**: Code quality checks
4. **Security**: Vulnerability scanning
5. **Build**: Docker image creation
6. **Deploy**: Image push to registry

### 3. Production Deployment
```bash
# Using Kubernetes
kubectl apply -f k8s/deployment.yaml

# Using Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

## 🛡️ Security Features

### Container Security
- Non-root user execution
- Minimal base image (Python slim)
- No unnecessary packages
- Regular security updates

### CI/CD Security
- Trivy vulnerability scanning
- Secret management via GitHub Secrets
- Dependency scanning
- SAST (Static Application Security Testing)

### Network Security
- Isolated Docker networks
- No unnecessary port exposure
- Internal service communication only

## 📈 Monitoring and Observability

### Metrics Collection
- **Prometheus**: Time-series metrics storage
- **Custom Metrics**: MOrA-specific application metrics
- **System Metrics**: Node and container metrics

### Visualization
- **Grafana**: Interactive dashboards
- **Pre-built Dashboards**: MOrA monitoring templates
- **Alerting**: Configurable alert rules

### Logging
- **Container Logs**: Docker Compose log aggregation
- **Application Logs**: Structured logging in MOrA
- **Centralized Logging**: Ready for ELK stack integration

## 🚀 Deployment Strategies

### 1. Blue-Green Deployment
- Zero-downtime deployments
- Instant rollback capability
- Traffic switching via load balancer

### 2. Canary Deployment
- Gradual traffic shifting
- A/B testing capabilities
- Risk mitigation

### 3. Rolling Updates
- Kubernetes native rolling updates
- Configurable update strategy
- Health check integration

## 🔄 Backup and Recovery

### Data Persistence
- **Models**: Persistent volume for trained models
- **Training Data**: Persistent volume for datasets
- **Reports**: Persistent volume for evaluation reports
- **Configuration**: ConfigMap and Secret management

### Backup Strategy
- **Automated Backups**: Cron job-based backups
- **Point-in-time Recovery**: Volume snapshots
- **Disaster Recovery**: Multi-region deployment

## 📋 Environment Management

### Development
- Local Docker Compose
- Hot reloading
- Debug mode enabled

### Staging
- Kubernetes cluster
- Production-like configuration
- Integration testing

### Production
- High availability setup
- Auto-scaling enabled
- Monitoring and alerting

## 🛠️ Maintenance

### Regular Tasks
- Security updates
- Dependency updates
- Performance monitoring
- Log rotation

### Troubleshooting
- Container health checks
- Log analysis
- Metrics investigation
- Network debugging

## 📚 Additional Resources

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure CI/CD passes
5. Submit a pull request

## 📞 Support

For DevOps-related issues:
- Check container logs: `docker-compose logs`
- Verify configuration: `docker-compose config`
- Test connectivity: `docker-compose exec mora-cli curl localhost:9090`
