#!/bin/bash
echo "stopping marqo..."
redis-cli shutdown 2>/dev/null || true
