# Reproduce experiments

# Arguments for each experiment
sensitivity_args=(
  "--app.dataset_type caida"
  "--app.repetitions 30"
  "--app.total_items 10000000"
  "--app.stream_size 10000000"
)

expansion_args=(
  "--app.dataset_type caida"
  "--app.repetitions 30"
  "--app.memory_increment_kb 8"
  "--app.initial_memory_kb 32"
  "--app.expansion_interval 100000"
  "--app.total_items 30000000"
  "--app.stream_size 20000000"
)

shrinking_args=(
  "--app.dataset_type caida"
  "--app.repetitions 30"
  "--app.initial_memory_kb 64"
  "--app.max_memory_kb 160"
  "--app.final_memory_kb 32"
  "--app.memory_decrement_kb 8"
  "--app.shrinking_interval 100000"
  "--app.total_items 3000000"
  "--app.stream_size 3000000"
)

merge_args=(
  "--app.dataset_type caida"
  "--app.repetitions 30"
  "--app.memory_budget_kb 32"
  "--app.stream_size 10000000"
)

split_args=(
  "--app.dataset_type caida"
  "--app.repetitions 30"
  "--app.memory_budget_kb 32"
  "--app.stream_size 10000000"
)

expansion_shrinking_args=(
  "--app.dataset_type caida"
  "--app.repetitions 30"
  "--app.m0_kb 32"
  "--app.m2_kb 16"
  "--app.memory_increment_kb 8"
  "--app.expansion_interval 100000"
  "--app.expansion_items 10000000"
  "--app.shrinking_items 3000000"
  "--app.stream_size 10000000"
)

dag_args=(
  "examples/dag/simple_dag.YAML"
)

run_and_viz() {
    local exp_name=$1
    local viz_name=$2
    shift 2
    local extra_args="$@"

    echo ""
    echo "Running ${exp_name}..."
    mkdir -p "$OUTPUT_DIR"
    build/release/bin/release/$exp_name $extra_args > "$OUTPUT_DIR/${exp_name}.txt" 2>&1

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
    echo "Finished reproducing ${viz_name} experiment"
}

DEFAULT_DAG_FILE="examples/dag/simple_dag.YAML"

help() {
  echo "Reproduce ReSketch evaluation benchmarks."
  echo
  echo "All experiments are run by default. To reproduce a specific one, pass "
  echo "the name using the syntax described below."
  echo
  echo "Syntax: $0 [<experiment>]"
  echo "Experiments:"
  echo -e "\tsensitivity: Sensitivity Analysis"
  echo -e "\texpand: Expansion performance"
  echo -e "\tshrink: Shrinking performance"
  echo -e "\tmerge: Merging accuracy"
  echo -e "\tsplit: Splitting accuracy"
  echo -e "\texpand_shrink: Combined expansion and shrinking"
  echo -e "\tdag: Run execution DAG"
}

if [ $# -eq 0 ]; then
  # If no arguments are passed, run all experiments
  exp="all"
elif [ $# -eq 1 ]; then
  # First argument, if present, is the experiment name
  exp="$1"
else
  help
  exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="output/reproduce_${exp}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Running reproduction experiments -> $OUTPUT_DIR"

# Activate Python venv if available
[ -d "venv" ] && source venv/bin/activate
[ -d ".venv" ] && source .venv/bin/activate

case "$exp" in
  all)
    run_and_viz "sensitivity_experiment" "sensitivity" ${sensitivity_args[@]} &
    run_and_viz "expansion_experiment" "expansion" ${expansion_args[@]} &
    run_and_viz "shrinking_experiment" "shrinking" ${shrinking_args[@]} &
    run_and_viz "expansion_shrinking_experiment" "expansion_shrinking" ${expansion_shrinking_args[@]} &
    run_and_viz "merge_experiment" "merge" ${merge_args[@]} &
    run_and_viz "split_experiment" "split" ${split_args[@]} &
    run_and_viz "dag_experiment" "dag" ${dag_args[@]} &
    ;;
  sensitivity)
    run_and_viz "sensitivity_experiment" "sensitivity" ${sensitivity_args[@]} &
    ;;
  expand)
    run_and_viz "expansion_experiment" "expansion" ${expansion_args[@]} &
    ;;
  shrink)
    run_and_viz "shrinking_experiment" "shrinking" ${shrinking_args[@]} &
    ;;
  merge)
    run_and_viz "merge_experiment" "merge" ${merge_args[@]} &
    ;;
  split)
    run_and_viz "split_experiment" "split" ${split_args[@]} &
    ;;
  expand_shrink)
    run_and_viz "expansion_shrinking_experiment" "expansion_shrinking" ${expansion_shrinking_args[@]} &
    ;;
  dag)
    run_and_viz "dag_experiment" "dag" ${dag_args[@]} &
    ;;
  *)
    echo "Invalid experiment ${exp}"
    help
    ;;
esac

wait
