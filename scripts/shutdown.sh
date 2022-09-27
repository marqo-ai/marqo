#!/bin/bash
echo "stopping marqo..."
docker stop marqo-os
kill "$(cat /var/run/docker.pid)"
rm /var/run/docker.pid
