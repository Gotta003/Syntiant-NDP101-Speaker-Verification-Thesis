#!/bin/bash

INPUT_FILE="temp file.txt"
OUTPUT_FILE="output_temp_file.txt"
ARRAY_NAME="scales"

#!/bin/bash

export LC_NUMERIC=C

echo "float ${ARRAY_NAME}[] = {" > "$OUTPUT_FILE"

# Alternative extraction using awk
awk '{
    # Find the number after colon and before asterisk
    split($0, parts, /[:*]/);
    gsub(/^[ \t]+|[ \t]+$/, "", parts[2]);  # trim whitespace
    printf "    %.20f,\n", parts[2]
}' "$INPUT_FILE" >> "$OUTPUT_FILE"

echo "};" >> "$OUTPUT_FILE"
