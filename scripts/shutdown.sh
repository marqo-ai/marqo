#!/bin/bash
echo "stopping marqo..."
docker stop marqo-os

if [ -z "$MARQO_ENABLE_THROTTLING" ]; then
    if [ "$MARQO_ENABLE_THROTTLING" != "FALSE" ]; then
        systemctl redis-server stop
        
kill "$(cat /var/run/docker.pid)"
rm /var/run/docker.pid
