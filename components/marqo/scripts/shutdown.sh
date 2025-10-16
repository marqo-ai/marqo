#!/bin/bash
echo "Stopping Marqo..."
kill -SIGINT $api_pid >/dev/null 2>&1 || true