#!/bin/bash

# Define the languages and their respective build directories
LANGUAGES=("en" "ko")
LANGUAGE_SWITCHER_FILE="book/_addons/language_switcher.js"
LANGUAGE_SELECTOR_CSS_FILE="book/_addons/language_selector.css"

# Main execution
echo "Starting pre-processing..."

# Copy addon files to the build directory
for i in "${!LANGUAGES[@]}"; do
    lang=${LANGUAGES[$i]}
    echo "Processing $lang..."
    dest_dir="book/${lang}"
    static_dest_dir="${dest_dir}/_static"
    mkdir -p "${static_dest_dir}"
    echo "Copying CSS files..."
    cp -f "$LANGUAGE_SELECTOR_CSS_FILE" "$static_dest_dir/"
    # Copy language_switcher.js to the _static directory
    echo "Copying language_switcher.js..."
    cp -f "$LANGUAGE_SWITCHER_FILE" "$static_dest_dir/"
done

echo "Pre-processing complete!"
