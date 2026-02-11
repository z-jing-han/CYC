DATA_ROOT="./Data"
ALL_DATA_DIR="./Data/all_data"

DEFAULT_BASE_DIR="./Data/Base_Input"
DEFAULT_START=1
DEFAULT_END=20


if [ "$1" == "-d" ]; then
    echo "Removing all V_ start directory in $DATA_ROOT"
    if [ -n "$DATA_ROOT" ]; then
        rm -rf "$DATA_ROOT"/V_*
    fi
    
    echo "Removing $ALL_DATA_DIR ..."
    rm -rf "$ALL_DATA_DIR"
    exit 0
fi

if [ "$#" -eq 0 ]; then
    BASE_INPUT_DIR="$DEFAULT_BASE_DIR"
    START_EXP="$DEFAULT_START"
    END_EXP="$DEFAULT_END"
    echo "No arguments provided. Using defaults:"
    echo "  Base Dir:  $BASE_INPUT_DIR"
    echo "  Range:     $START_EXP to $END_EXP"
elif [ "$#" -eq 3 ]; then
    BASE_INPUT_DIR="$1"
    START_EXP="$2"
    END_EXP="$3"
else
    echo "Usage:"
    echo "  Default run: $0  (Equivalent to: $0 $DEFAULT_BASE_DIR $DEFAULT_START $DEFAULT_END)"
    echo "  Custom run:  $0 [Base_dir] [start_exp_V] [end_exp_V]"
    echo "  Clean data:  $0 -d"
    exit 1
fi

if [ ! -d "$BASE_INPUT_DIR" ]; then
    echo "Error: Base directory '$BASE_INPUT_DIR' does not exist."
    exit 1
fi

if [ ! -f "$BASE_INPUT_DIR/config.json" ]; then
    echo "Error: Can't find $BASE_INPUT_DIR/config.json"
    exit 1
fi

mkdir -p "$ALL_DATA_DIR/competitor/Carbon_Emission"
mkdir -p "$ALL_DATA_DIR/competitor/Queue_Len"
mkdir -p "$ALL_DATA_DIR/dwpa/Carbon_Emission"
mkdir -p "$ALL_DATA_DIR/dwpa/Queue_Len"

copy_result() {
    local v_tag=$1      
    local algo=$2      
    local metric=$3    
    local src_file=$4  
    local dest_prefix=$5 

    local src_path="$DATA_ROOT/V_${v_tag}_Output/figures/$algo/$metric/$src_file"
    local dest_path="$ALL_DATA_DIR/$algo/$metric/${dest_prefix}_V_${v_tag}.png"

    if [ -f "$src_path" ]; then
        cp "$src_path" "$dest_path"
    fi
}

for (( exp=START_EXP; exp<=END_EXP; exp++ )); do
    for digit in {1..9}; do
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

        ./test.sh "./Data/$CURRENT_INPUT_NAME" "./Data/$CURRENT_OUTPUT_NAME"

        copy_result "$V_TAG" "competitor" "Carbon_Emission" "Total_carbon.png" "Total_carbon"
        copy_result "$V_TAG" "competitor" "Queue_Len" "Total_queue.png" "Total_queue"
        copy_result "$V_TAG" "dwpa" "Carbon_Emission" "Total_carbon.png" "Total_carbon"
        copy_result "$V_TAG" "dwpa" "Queue_Len" "Total_queue.png" "Total_queue"

    done
done

echo "Finish ALL Task"