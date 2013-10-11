#!/bin/bash

cd /Users/analysis/
. .profile
watch_analysis.sh /Users/analysis/Python/WatchFolder/Full/ --n_jobs=4 --verbose=1 --full
