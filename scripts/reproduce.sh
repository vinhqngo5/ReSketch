# Reproduce experiments
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="output/reproduce_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Running reproduction experiments -> $OUTPUT_DIR"

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
    ./build/release/bin/release/$exp_name $extra_args > "$OUTPUT_DIR/${exp_name}.txt" 2>&1
    
    # Find the most recent JSON file for this experiment
    local json_file=$(ls -t output/${viz_name}_results_*.json 2>/dev/null | head -1)
    
    if [ -n "$json_file" ] && [ -f "$json_file" ]; then
        mkdir -p "$OUTPUT_DIR"
        
        # Move to output directory with clean name
        local new_name="$OUTPUT_DIR/${viz_name}_results.json"
        if mv "$json_file" "$new_name" 2>/dev/null; then
            echo "Visualizing ${viz_name}..."
            python "scripts/visualize_${viz_name}.py" --input "$new_name" --output "$OUTPUT_DIR/$viz_name" --show-within-variance 2>&1
        else
            echo "Warning: Failed to move JSON file for ${viz_name}"
        fi
    else
        echo "Warning: No JSON output found for ${viz_name}"
    fi
}


run_and_viz "sensitivity_experiment" "sensitivity" \
    --app.dataset_type caida \
    --app.repetitions 30 \
    --app.memory_budget_kb 32 \
    --app.total_items 10000000 \
    --app.stream_size 10000000 &

run_and_viz "expansion_experiment" "expansion" \
    --app.repetitions 30 \
    --app.dataset_type caida \
    --app.memory_increment_kb 8 \
    --app.initial_memory_kb 32 \
    --app.expansion_interval 100000 \
    --app.total_items 30000000 \
    --app.stream_size 20000000 &

run_and_viz "shrinking_experiment" "shrinking" \
    --app.repetitions 30 \
    --app.dataset_type caida \
    --app.initial_memory_kb 64 \
    --app.max_memory_kb 160 \
    --app.final_memory_kb 32 \
    --app.memory_decrement_kb 8 \
    --app.shrinking_interval 100000 \
    --app.total_items 3000000 \
    --app.stream_size 3000000 &

run_and_viz "merge_experiment" "merge" \
    --app.dataset_type caida \
    --app.repetitions 30 \
    --app.memory_budget_kb 32 \
    --app.stream_size 10000000 &

run_and_viz "split_experiment" "split" \
    --app.dataset_type caida \
    --app.repetitions 30 \
    --app.memory_budget_kb 32 \
    --app.stream_size 10000000 &

run_and_viz "dag_experiment" "dag" "examples/dag/simple_dag.YAML" &

# Wait for all background jobs to complete
wait

echo ""
echo "Done: $OUTPUT_DIR"