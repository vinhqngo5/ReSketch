#!/bin/bash
# Initialize submodules and apply local patches

set -e

SUBMODULE_PATH="src/frequency_summary/geometric_sketch"
PATCH_DIR="patches/geometric_sketch"
LATEST_PATCH="$PATCH_DIR/submodule_changes_latest.patch"

echo "Setting up submodules..."

git submodule init
git submodule update

echo "Submodules initialized"

if [ -f "$LATEST_PATCH" ]; then
    echo ""
    echo "Applying local patches..."
    
    cd "$SUBMODULE_PATH"
    
    if git apply --check "../../../$LATEST_PATCH" 2>/dev/null; then
        git apply "../../../$LATEST_PATCH"
        echo "Patches applied"
        
        echo ""
        echo "Changes:"
        git diff --stat
    else
        echo "Warning: Patch cannot be applied cleanly"
        echo "The submodule may have been updated upstream"
        echo "You may need to manually resolve conflicts or regenerate patches"
        exit 1
    fi
    
    cd - > /dev/null
else
    echo ""
    echo "No patches found at $LATEST_PATCH"
    echo "If you have local modifications, run: ./scripts/create_submodule_patches.sh"
fi

echo ""
echo "Setup complete"
