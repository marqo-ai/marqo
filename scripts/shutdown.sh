#!/bin/bash
echo "stopping marqo..."
redis-cli shutdown 2>/dev/null || true
/opt/vespa/bin/vespa-stop-services >/dev/null 2>&1
