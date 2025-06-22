#!/bin/bash

# Define arguments for the sample command
MECH_PATH="/home/xk/Software/5_dfode_kit/Burke2012_s9r23.yaml"
CASE_PATH="/home/xk/Software/5_dfode_kit/DFODE-kit/cases/oneD_freely_propagating_flame"
SAVE_PATH="/home/xk/Software/5_dfode_kit/DFODE-kit/cases/oneD_freely_propagating_flame/flame_data.h5"

# Execute the sample command
dfode-kit sample --mech "$MECH_PATH" --case "$CASE_PATH" --save "$SAVE_PATH" --include_mesh

# Define arguments for the h52npy command
SOURCE_FILE="$SAVE_PATH"
OUTPUT_FILE="test.npy"

# Execute the h52npy command
dfode-kit h52npy --source "$SOURCE_FILE" --save_to "$OUTPUT_FILE"

# Define arguments for the label command
LABEL_SOURCE="/home/xk/Software/5_dfode_kit/DFODE-kit/cases/oneD_freely_propagating_flame/test.npy"
LABEL_OUTPUT="test_labeled.npy"

# Execute the label command
dfode-kit label --mech "$MECH_PATH" --time '(1e-8 1e-7)' --source "$LABEL_SOURCE" --save "$LABEL_OUTPUT"