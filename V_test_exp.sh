#!/bin/bash

DATA_ROOT="./Data"
ALL_DATA_DIR="./Data/all_data"

DEFAULT_BASE_DIR="./Data/Base_Input"
TRADEOFF_OUTPUT_DIR="$DATA_ROOT/Tradeoff_Results"
DEFAULT_START_EXP=19
DEFAULT_START_DIGIT=6
DEFAULT_END_EXP=21
DEFAULT_END_DIGIT=7

if [ "$1" == "-d" ]; then
    echo "Removing all V_ start directory in $DATA_ROOT"
    if [ -n "$DATA_ROOT" ]; then
        rm -rf "$DATA_ROOT"/V_*
    fi
    
    echo "Removing $ALL_DATA_DIR ..."
    rm -rf "$ALL_DATA_DIR"

    echo "Removing $TRADEOFF_OUTPUT_DIR ..."
    rm -rf "$TRADEOFF_OUTPUT_DIR"

    exit 0
fi

if [ "$#" -eq 0 ]; then
    BASE_INPUT_DIR="$DEFAULT_BASE_DIR"
    START_EXP="$DEFAULT_START_EXP"
    START_DIGIT="$DEFAULT_START_DIGIT"
    END_EXP="$DEFAULT_END_EXP"
    END_DIGIT="$DEFAULT_END_DIGIT"
    echo "No arguments provided. Using defaults:"
    echo "  Base Dir:    $BASE_INPUT_DIR"
    echo "  Range:       ${START_DIGIT}e${START_EXP} to ${END_DIGIT}e${END_EXP}"
elif [ "$#" -eq 5 ]; then
    BASE_INPUT_DIR="$1"
    START_EXP="$2"
    START_DIGIT="$3"
    END_EXP="$4"
    END_DIGIT="$5"
else
    echo "Usage:"
    echo "  Default run: $0"
    echo "  Custom run:  $0 [Base_dir] [start_exp] [start_digit] [end_exp] [end_digit]"
    echo "  Clean data:  $0 -d"
    exit 1
fi

if [ ! -d "$BASE_INPUT_DIR" ]; then
    echo "Error: Base directory '$BASE_INPUT_DIR' does not exist."
    exit 1
fi

copy_result() {
    local v_tag=$1
    local algo=$2
    local metric=$3
    local src_file=$4
    local dest_prefix=$5

    local src_path="$DATA_ROOT/V_${v_tag}_Output/figures/$algo/$metric/$src_file"
    local dest_path="$ALL_DATA_DIR/$algo/$metric/${dest_prefix}_V_${v_tag}.png"

    if [ -f "$src_path" ]; then
        mkdir -p "$(dirname "$dest_path")"
        cp "$src_path" "$dest_path"
    fi
}

for (( exp=START_EXP; exp<=END_EXP; exp++ )); do
    
    if [ "$START_EXP" -eq "$END_EXP" ]; then
        d_start=$START_DIGIT
        d_end=$END_DIGIT
    else
        if [ "$exp" -eq "$START_EXP" ]; then
            d_start=$START_DIGIT
            d_end=9
        elif [ "$exp" -eq "$END_EXP" ]; then
            d_start=1
            d_end=$END_DIGIT
        else
            d_start=1
            d_end=9
        fi
    fi

    for (( digit=d_start; digit<=d_end; digit++ )); do
        if [ "$exp" -eq 0 ]; then
            V_VAL="${digit}"
        else
            V_VAL="${digit}e${exp}"
        fi

        V_TAG=$(printf "E%02d_D%d" "$exp" "$digit")

        echo "Handle V = $V_VAL ($V_TAG) ..."

        CURRENT_INPUT_NAME="V_${V_TAG}_Input"
        CURRENT_OUTPUT_NAME="V_${V_TAG}_Output"
        CURRENT_INPUT_PATH="$DATA_ROOT/$CURRENT_INPUT_NAME"
        
        rm -rf "$CURRENT_INPUT_PATH"
        cp -r "$BASE_INPUT_DIR" "$CURRENT_INPUT_PATH"

        TARGET_CONFIG="$CURRENT_INPUT_PATH/config.json"
        tmp_json=$(mktemp)
        
        if jq --argjson v "$V_VAL" '.system_settings.trade_off_V = $v' "$TARGET_CONFIG" > "$tmp_json"; then
                mv "$tmp_json" "$TARGET_CONFIG"
        else
            echo "Error: jq failed to update config for V=$V_VAL"
            rm "$tmp_json"
            exit 1
        fi

        ./run.sh "./Data/$CURRENT_INPUT_NAME" "./Data/$CURRENT_OUTPUT_NAME"

        copy_result "$V_TAG" "competitor" "Carbon_Emission" "Total_carbon.png" "Total_carbon"
        copy_result "$V_TAG" "competitor" "Queue_Len" "Total_queue.png" "Total_queue"
        copy_result "$V_TAG" "dwpa" "Carbon_Emission" "Total_carbon.png" "Total_carbon"
        copy_result "$V_TAG" "dwpa" "Queue_Len" "Total_queue.png" "Total_queue"

    done
done

echo "Finish ALL Task"

echo "Generating Trade-off plots..."
mkdir -p "$TRADEOFF_OUTPUT_DIR"

python plot.py --plot_tradeoff --tradeoff_data_dir "$DATA_ROOT" --output_dir "$TRADEOFF_OUTPUT_DIR"