#!/bin/bash

CURRENT_DIR=$(pwd)

cd "$(dirname "$0")/.."

echo "Running training script..."
python3 scripts/run_training.py

if [ $? -ne 0 ]; then
    echo "run_training.py failed. Exiting..."
    exit 1
fi

echo "Pausing for 1 minute..."
sleep 60

echo "Running evaluation script..."
python3 scripts/run_evaluation.py

if [ $? -ne 0 ]; then
    echo "run_evaluation.py failed. Exiting..."
    exit 1
fi

echo "Running visualization script..."
python3 scripts/visualize_results.py

if [ $? -ne 0 ]; then
    echo "visualize_results.py failed."
    exit 1
fi

echo "All scripts executed successfully."

cd "$CURRENT_DIR"
