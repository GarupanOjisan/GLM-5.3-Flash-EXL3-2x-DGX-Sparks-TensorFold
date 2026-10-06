#!/bin/bash
# static server for the dashboard and the metric files, LAN only
cd ~/glm53-metrics && exec python3 -m http.server 8090 --bind 0.0.0.0 >> ~/glm53-metrics/serve.log 2>&1
