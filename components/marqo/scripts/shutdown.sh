#!/bin/bash
echo "Stopping Marqo..."
kill -SIGINT $api_pid >/dev/null 2>&1 || true
redis-cli shutdown >/dev/null 2>&1 || true
/opt/vespa/bin/vespa-stop-services >/dev/null 2>&1 || true
