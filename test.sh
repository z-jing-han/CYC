#!/bin/bash

INPUT_DIR="${1:-Data/Base_Input/}"
OUTPUT_DIR="${2:-Data/Base_Output/}"

if [ ! -f "${INPUT_DIR}/data_arrival.csv" ]; then
    python3 generate.py --dir "${INPUT_DIR}"
fi

python3 main.py --input_dir "${INPUT_DIR}" --output_dir "${OUTPUT_DIR}"
python3 plot.py --input_dir "${INPUT_DIR}" --output_dir "${OUTPUT_DIR}"