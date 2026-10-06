#!/bin/bash
# one line per minute: unix ts, GPU sm clock MHz, power W, temperature C, host memory used/free MiB, throttle reasons
d=$(date +%Y-%m-%d); g=$(nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') || exit 0
m=$(free -m | awk 'NR==2{print $3","$7}')
echo "$(date +%s),$g,$m" >> ~/glm53-metrics/gpu-$d.csv
