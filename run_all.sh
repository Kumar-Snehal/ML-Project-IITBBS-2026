#!/bin/sh
# Runs every pipeline step in order. Run from the project root:  sh run_all.sh
set -e
python scripts/01_clean_data.py
python scripts/02_build_features.py
python scripts/check_leakage.py
python scripts/03_split_data.py
python scripts/04_analyse_features.py
