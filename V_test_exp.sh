#!/bin/bash

DATA_ROOT="./Data"
ALL_DATA_DIR="./Data/all_data"

DEFAULT_BASE_INPUT="./Data/Thesis_Input"
DEFAULT_BASE_OUTPUT="./Data/Thesis_Output"
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
    BASE_INPUT_DIR="$DEFAULT_BASE_INPUT"
    BASE_OUTPUT_DIR="$DEFAULT_BASE_OUTPUT"
    START_EXP="$DEFAULT_START_EXP"
    START_DIGIT="$DEFAULT_START_DIGIT"
    END_EXP="$DEFAULT_END_EXP"
    END_DIGIT="$DEFAULT_END_DIGIT"
    echo "No arguments provided. Using defaults:"
    echo "  Base Input:  $BASE_INPUT_DIR"
    echo "  Base Output: $BASE_OUTPUT_DIR"
    echo "  Range:       ${START_DIGIT}e${START_EXP} to ${END_DIGIT}e${END_EXP}"
elif [ "$#" -eq 6 ]; then
    BASE_INPUT_DIR="$1"
    BASE_OUTPUT_DIR="$2"
    START_EXP="$3"
    START_DIGIT="$4"
    END_EXP="$5"
    END_DIGIT="$6"
else
    echo "Usage:"
    echo "  Default run: $0"
    echo "  Custom run:  $0 [Base_input_dir] [Base_output_dir] [start_exp] [start_digit] [end_exp] [end_digit]"
    echo "  Clean data:  $0 -d"
    exit 1
fi

if [ ! -d "$BASE_INPUT_DIR" ]; then
    echo "Error: Base input directory '$BASE_INPUT_DIR' does not exist."
    exit 1
fi

if [ ! -d "$BASE_OUTPUT_DIR" ]; then
    echo "Error: Base output directory '$BASE_OUTPUT_DIR' does not exist. Cannot copy .pth weights."
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
        CURRENT_OUTPUT_PATH="$DATA_ROOT/$CURRENT_OUTPUT_NAME"
        
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

        rm -rf "$CURRENT_OUTPUT_PATH"
        mkdir -p "$CURRENT_OUTPUT_PATH"
        
        # find "$BASE_OUTPUT_DIR" -type f -name "*.pth" | while read -r pth_file; do
        #     rel_path="${pth_file#$BASE_OUTPUT_DIR/}"
        #     mkdir -p "$CURRENT_OUTPUT_PATH/$(dirname "$rel_path")"
        #     cp "$pth_file" "$CURRENT_OUTPUT_PATH/$rel_path"
        # done

        ./run.sh "./Data/$CURRENT_INPUT_NAME" "./Data/$CURRENT_OUTPUT_NAME"

        echo "Copying result figures..."
        for algo_group in "competitor" "dwpa" "MAAO" "MA" "MATWO"; do
            copy_result "$V_TAG" "$algo_group" "Carbon_Emission" "Total_carbon.png" "Total_carbon"
            copy_result "$V_TAG" "$algo_group" "Queue_Len" "Total_queue.png" "Total_queue"
        done

    done
done

echo "Finish ALL Task"

echo "Generating Trade-off plots..."
mkdir -p "$TRADEOFF_OUTPUT_DIR"

python plot.py --plot_tradeoff --tradeoff_data_dir "$DATA_ROOT" --output_dir "$TRADEOFF_OUTPUT_DIR"