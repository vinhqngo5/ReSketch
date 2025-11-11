#!/bin/bash
# Create patch files from submodule modifications

set -e

SUBMODULE_PATH="src/frequency_summary/geometric_sketch"
PATCH_DIR="patches/geometric_sketch"

echo "Creating patches from submodule changes..."

if [ ! -d "$SUBMODULE_PATH" ]; then
    echo "Error: Submodule not found at $SUBMODULE_PATH"
    exit 1
fi

mkdir -p "$PATCH_DIR"

cd "$SUBMODULE_PATH"

if ! git diff --quiet HEAD; then
    echo "Found uncommitted changes"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    PATCH_FILE="../../../$PATCH_DIR/submodule_changes_${TIMESTAMP}.patch"
    
    git diff HEAD > "$PATCH_FILE"
    echo "Patch created: $PATCH_FILE"
    
    LATEST_PATCH="../../../$PATCH_DIR/submodule_changes_latest.patch"
    cp "$PATCH_FILE" "$LATEST_PATCH"
    echo "Latest patch updated: $LATEST_PATCH"
    
    echo ""
    git diff HEAD --stat
else
    echo "No changes found"
fi

cd - > /dev/null

echo ""
echo "Done. Add patches to git: git add $PATCH_DIR/"
