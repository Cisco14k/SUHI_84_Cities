#!/bin/bash
cd /home/claude/n84
python3 src/lagr84.py all 2020 2020 >> logs/lagr.log 2>&1
python3 src/lagr84.py 3,0,11,19,21,12,31,28,34,36,39,46,56,48,59,64,62,65,74,73,77 2016 2019 >> logs/lagr.log 2>&1
python3 src/lagr84.py all 2019 2019 >> logs/lagr.log 2>&1
python3 src/lagr84.py all 2018 2018 >> logs/lagr.log 2>&1
