#!/bin/sh
# Runs every pipeline step in order. Run from the project root:  sh run_all.sh
# Set PYTHON to use a specific interpreter, e.g.  PYTHON=.venv/bin/python sh run_all.sh
set -e
PY=${PYTHON:-python3}
$PY -m src.cleaning.clean_data
$PY -m src.features.build_features
$PY -m src.features.check_leakage
$PY -m src.features.split_data
$PY -m src.features.analyse_features
$PY -m src.models.naive_bayes
$PY -m src.models.logistic_regression
$PY -m src.models.evaluate
