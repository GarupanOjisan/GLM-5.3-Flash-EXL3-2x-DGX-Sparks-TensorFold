#!/bin/bash
# one /health sample per minute, as a JSON line with a unix timestamp; one file per day
d=$(date +%Y-%m-%d); j=$(curl -s -m 5 http://localhost:8000/health) || exit 0
[ -n "$j" ] && printf "{\"ts\": %s, \"h\": %s}\n" "$(date +%s)" "$j" >> ~/glm53-metrics/health-$d.jsonl
