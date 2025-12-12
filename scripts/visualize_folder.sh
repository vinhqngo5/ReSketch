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

# Visualize all experiment types
visualize_if_exists "expansion"
visualize_if_exists "shrinking"
visualize_if_exists "merge"
visualize_if_exists "split"
visualize_if_exists "sensitivity"
visualize_if_exists "dag"

echo ""
echo "Done visualizing: $FOLDER"
