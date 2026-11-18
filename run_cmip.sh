#!/bin/bash
cd /home/claude/n84
for m in "$@"; do python3 src/cmip.py $m >> logs/cmip_$1.log 2>&1; done
