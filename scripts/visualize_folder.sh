# Visualize all JSON results in a given folder
FOLDER=$1

if [ ! -d "$FOLDER" ]; then
    echo "Error: Directory '$FOLDER' does not exist"
    exit 1
fi

echo "Visualizing results in: $FOLDER"

# Activate Python venv if available
[ -d "venv" ] && source venv/bin/activate
[ -d ".venv" ] && source .venv/bin/activate

# Function to visualize a specific experiment type
visualize_if_exists() {
    local exp_name=$1
    local json_pattern="${FOLDER}/${exp_name}_results.json"
    
    if [ -f "$json_pattern" ]; then
        echo "Visualizing ${exp_name}..."
        python "scripts/visualize_${exp_name}.py" --input "$json_pattern" --output "${FOLDER}/${exp_name}" --show-within-variance 2>&1
        if [ $? -eq 0 ]; then
            echo "${exp_name} visualization complete"
        else
            echo "${exp_name} visualization failed"
        fi
    else
        echo "No ${exp_name}_results.json found, skipping"
    fi
}

visualize_if_exists_with_plot_every() {
    local exp_name=$1
    local plot_every=$2
    local json_pattern="${FOLDER}/${exp_name}_results.json"
    
    if [ -f "$json_pattern" ]; then
        echo "Visualizing ${exp_name}..."
        python "scripts/visualize_${exp_name}.py" --input "$json_pattern" --output "${FOLDER}/${exp_name}" --show-within-variance --plot-every "$plot_every" 2>&1
        if [ $? -eq 0 ]; then
            echo "${exp_name} visualization complete"
        else
            echo "${exp_name} visualization failed"
        fi
    else
        echo "No ${exp_name}_results.json found, skipping"
    fi
}

visualize_if_exists_with_plot_every "expansion" 20
visualize_if_exists_with_plot_every "shrinking" 4
visualize_if_exists_with_plot_every "expansion_shrinking" 4
visualize_if_exists "merge"
visualize_if_exists "split"

# Sensitivity with specific memory budget
if [ -f "${FOLDER}/sensitivity_results.json" ]; then
    echo "Visualizing sensitivity..."
    python "scripts/visualize_sensitivity_single.py" --input "${FOLDER}/sensitivity_results.json" --output "${FOLDER}/sensitivity" --show-within-variance --memory-budget 64 2>&1
    if [ $? -eq 0 ]; then
        echo "sensitivity visualization complete"
    else
        echo "sensitivity visualization failed"
    fi
else
    echo "No sensitivity_results.json found, skipping"
fi

visualize_if_exists "dag"

echo ""
echo "Done visualizing: $FOLDER"
