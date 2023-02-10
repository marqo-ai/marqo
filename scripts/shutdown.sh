#!/bin/bash
echo "stopping marqo..."
docker stop marqo-os
redis-cli shutdown 2>/dev/null || true
kill "$(cat /var/run/docker.pid)"
rm /var/run/docker.pid
