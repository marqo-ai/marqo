#!/bin/bash
# This script is meant to be run at buildtime.

if [ -z "$MARQO_ENABLE_THROTTLING" ]; then
    if [ "$MARQO_ENABLE_THROTTLING" != "FALSE" ]; then
        apt-get install redis-server -y