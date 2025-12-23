run_and_viz() {
    local exp_name=$1
    local viz_name=$2
    shift 2
    local extra_args="$@"
    
    echo ""
    echo "Running ${exp_name} (${OUTPUT_DIR})..."
    mkdir -p "$OUTPUT_DIR"
    build/release/bin/release/$exp_name $extra_args --app.repetitions 5 > "$OUTPUT_DIR/${exp_name}.txt" 2>&1
    
    # Find the most recent JSON file for this experiment
    local json_file=$(ls -t output/${viz_name}_results_*.json 2>/dev/null | head -1)
    
    if [ -n "$json_file" ] && [ -f "$json_file" ]; then
        mkdir -p "$OUTPUT_DIR"

        # Move to output directory with clean name
        local new_name="$OUTPUT_DIR/${viz_name}_results.json"
        if mv "$json_file" "$new_name" 2>/dev/null; then
            echo "Visualizing ${viz_name}..."
            python "scripts/visualize_${viz_name}.py" --input "$new_name" --output "$OUTPUT_DIR/$viz_name" 2>&1
        else
            echo "Warning: Failed to move JSON file for ${viz_name}"
        fi
    else
        echo "Warning: No JSON output found for ${viz_name}"
    fi

    echo
    echo "Finished ${exp_name}"
}

DEFAULT_DAG_FILE="examples/dag/simple_dag.YAML"

help() {
  echo "Run ReSketch experiments with default settings."
  echo
  echo "Syntax: $0 <experiment> [<dag_path>]"
  echo "Experiments:"
  echo -e "\tsensitivity: Sensitivity Analysis"
  echo -e "\texpand: Expansion performance"
  echo -e "\tshrink: Shrinking performance"
  echo -e "\tmerge: Merging accuracy"
  echo -e "\tsplit: Splitting accuracy"
  echo -e "\tdag: Run execution DAG (default: ${DEFAULT_DAG_FILE}, or pass a path as next argument)"
  echo -e "\tall: Run all above experiments"
}

# If no arguments are passed, show help
if [ $# -eq 0 ]; then
  help
  exit 1
fi

# First argument is the experiment name
exp="$1"

# Second argment is the dag file path, or the default if not present
dag_file="${2:-$DEFAULT_DAG_FILE}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="output/run_${exp}_${TIMESTAMP}"

# Activate Python venv if available
[ -d "venv" ] && source venv/bin/activate
[ -d ".venv" ] && source .venv/bin/activate

case "$exp" in
  all)
    run_and_viz "sensitivity_experiment" "sensitivity" &
    run_and_viz "expansion_experiment" "expansion" &
    run_and_viz "shrinking_experiment" "shrinking" &
    run_and_viz "merge_experiment" "merge" &
    run_and_viz "split_experiment" "split" &
    run_and_viz "dag_experiment" "dag" $dag_file &
    ;;
  sensitivity)
    run_and_viz "sensitivity_experiment" "sensitivity" &
    ;;
  expand)
    run_and_viz "expansion_experiment" "expansion" &
    ;;
  shrink)
    run_and_viz "shrinking_experiment" "shrinking" &
    ;;
  merge)
    run_and_viz "merge_experiment" "merge" &
    ;;
  split)
    run_and_viz "split_experiment" "split" &
    ;;
  dag)
    run_and_viz "dag_experiment" "dag" $dag_file &
    ;;
  *)
    echo "Invalid experiment ${exp}"
    help
    ;;
esac

wait
