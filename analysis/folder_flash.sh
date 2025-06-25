#!/bin/bash

# Check if folder path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <folder_path>"
    exit 1
fi

folder_path="$1"

# Find all .wav files in the folder
wav_files=("$folder_path"/*.wav)

# Check if any .wav files exist
if [ ${#wav_files[@]} -eq 0 ]; then
    echo "No .wav files found in $folder_path"
    exit 1
fi

# Create a temporary script file
temp_script="ndp_flash.sh"
chmod +x "$temp_script"

# Process each .wav file
count=0
for wav_file in "${wav_files[@]}"; do
    echo "Processing file: $wav_file"
    "./$temp_script" "$wav_file" "$count"
    ((count++))
    echo ""
    echo "========================================="
    echo "Finished $count out ${#wav_files[@]}"
    echo "========================================="
    echo ""
done
