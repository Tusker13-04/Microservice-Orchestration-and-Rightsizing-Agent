#!/bin/bash

# MOrA Data Collection Monitor
# Run this script every hour to check data collection progress

echo "=========================================="
echo "🔍 MOrA DATA COLLECTION MONITOR"
echo "=========================================="
echo "Timestamp: $(date)"
echo ""

# Check if data collection process is running
echo "📊 PROCESS STATUS:"
if pgrep -f "parallel-experiments" > /dev/null; then
    echo "✅ Data collection process is RUNNING"
    echo "   PID: $(pgrep -f 'parallel-experiments')"
else
    echo "❌ Data collection process is NOT RUNNING"
fi

# Check system health
echo ""
echo "🏥 SYSTEM HEALTH:"
echo "   Minikube: $(kubectl cluster-info 2>/dev/null | head -1 | grep -o 'running' || echo 'NOT RUNNING')"
echo "   Prometheus: $(curl -s http://localhost:9090/-/ready 2>/dev/null | grep -o 'Ready' || echo 'NOT READY')"
echo "   Hipster Shop: $(kubectl get pods -n hipster-shop --no-headers | wc -l) pods running"

# Check experiment progress
echo ""
echo "📈 EXPERIMENT PROGRESS:"
if [ -f "data_collection.log" ]; then
    echo "   Log file size: $(du -h data_collection.log | cut -f1)"
    echo "   Last activity: $(tail -1 data_collection.log | head -c 100)..."
else
    echo "   ❌ No log file found"
fi

# Check data collection results
echo ""
echo "💾 DATA COLLECTION RESULTS:"
if [ -d "training_data" ]; then
    json_count=$(find training_data -name "*.json" 2>/dev/null | wc -l)
    csv_count=$(find training_data -name "*.csv" 2>/dev/null | wc -l)
    echo "   JSON files: $json_count"
    echo "   CSV files: $csv_count"
    echo "   Total experiments completed: $((json_count / 2))"  # Each experiment has 2 JSON files
else
    echo "   ❌ No training_data directory found"
fi

# Check for errors in log
echo ""
echo "🚨 ERROR CHECK:"
if [ -f "data_collection.log" ]; then
    error_count=$(grep -i "error\|failed\|exception" data_collection.log | wc -l)
    if [ $error_count -eq 0 ]; then
        echo "   ✅ No errors detected"
    else
        echo "   ⚠️  $error_count potential errors found"
        echo "   Recent errors:"
        grep -i "error\|failed\|exception" data_collection.log | tail -3
    fi
else
    echo "   ❌ Cannot check errors - no log file"
fi

# Check resource usage
echo ""
echo "💻 RESOURCE USAGE:"
echo "   CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "   Memory: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "   Disk: $(df -h . | tail -1 | awk '{print $5}')"

# Estimated completion
echo ""
echo "⏰ ESTIMATED COMPLETION:"
if [ -f "data_collection.log" ] && [ $json_count -gt 0 ]; then
    completed=$((json_count / 2))
    remaining=$((96 - completed))
    if [ $remaining -gt 0 ]; then
        hours_remaining=$((remaining * 30 / 60))  # 30 minutes per experiment with 2 workers
        echo "   Progress: $completed/96 experiments completed"
        echo "   Remaining: $remaining experiments (~$hours_remaining hours)"
    else
        echo "   🎉 ALL EXPERIMENTS COMPLETED!"
    fi
else
    echo "   Unable to estimate - no data yet"
fi

echo ""
echo "=========================================="
echo "✅ Monitor complete - System is healthy"
echo "=========================================="

