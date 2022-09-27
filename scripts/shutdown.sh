#!/bin/bash
echo "stopping marqo..."
docker stop marqo-os
ps axf | grep docker | grep -v grep | awk '{print "kill " $1}' | sh
exit
