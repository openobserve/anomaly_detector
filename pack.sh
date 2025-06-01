#!/bin/sh

# This script is used to pack the Action

# Get the current directory
CURRENT_DIR=$(pwd)

# Get the Action name
ACTION_NAME=anomaly_detector

# Pack the Action
zip -r $ACTION_NAME.zip pyproject.toml rrcf_model.pkl main.py



