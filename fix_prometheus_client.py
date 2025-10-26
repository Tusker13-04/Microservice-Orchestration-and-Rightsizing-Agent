#!/usr/bin/env python3
import re

# Read the file
with open('src/mora/monitoring/prometheus_client.py', 'r') as f:
    content = f.read()

# Fix method signatures
methods_to_fix = [
    '_get_network_rx_bytes',
    '_get_network_tx_bytes', 
    '_get_cpu_throttled',
    '_get_pod_restarts'
]

for method in methods_to_fix:
    # Fix method signature
    pattern = f'def {method}\\(self, pod_selector: str, start_time: datetime, end_time: datetime\\) -> pd\\.DataFrame:'
    replacement = f'def {method}(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:'
    content = re.sub(pattern, replacement, content)
    
    # Add pod_selector definition after method signature
    pattern = f'(def {method}\\(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime\\) -> pd\\.DataFrame:\n        """.*?"""\n)'
    replacement = r'\1        pod_selector = f\'pod=~"{service_name}.*",namespace="{namespace}"\'\n'
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write the fixed content
with open('src/mora/monitoring/prometheus_client.py', 'w') as f:
    f.write(content)

print("✅ All method signatures fixed")
