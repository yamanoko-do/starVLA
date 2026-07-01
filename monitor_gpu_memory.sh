#!/bin/bash
# Monitor GPU memory usage for training

echo "Starting GPU memory monitoring..."
echo "Logging to: /tmp/gpu_memory_monitor.log"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    GPU_INFO=$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,nounit,nounits | head -n +1)

    # Get detailed memory info
    MEMORY_INFO=$(nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,nounit,nounits)

    echo "[$TIMESTAMP]"
    echo "$GPU_INFO"
    echo "$MEMORY_INFO"
    echo "-------------------"

    sleep 5
done >> /tmp/gpu_memory_monitor.log 2>&1
