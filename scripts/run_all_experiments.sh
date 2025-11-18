# Run all experiments and generate visualizations
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="output/run_all_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Running experiments -> $OUTPUT_DIR"

# Activate Python venv if available
[ -d "venv" ] && source venv/bin/activate
[ -d ".venv" ] && source .venv/bin/activate

run_and_viz() {
    local exp_name=$1
    local viz_name=$2
    shift 2
    local extra_args="$@"
    
    echo ""
    echo "Running ${exp_name}..."
    build/release/bin/release/$exp_name $extra_args --app.repetitions 2 > "$OUTPUT_DIR/${exp_name}.txt" 2>&1
    
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
}

run_and_viz "expansion_experiment" "expansion" &
run_and_viz "merge_experiment" "merge" &
run_and_viz "split_experiment" "split" &
run_and_viz "shrinking_experiment" "shrinking" &
run_and_viz "sensitivity_experiment" "sensitivity" &
run_and_viz "dag_experiment" "dag" "examples/dag/simple_dag.YAML" &

wait

echo ""
echo "Done: $OUTPUT_DIR"
