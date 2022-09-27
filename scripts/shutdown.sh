#!/bin/bash
docker stop marqo-os
ps axf | grep docker | grep -v grep | awk '{print "kill " $1}' | sh
