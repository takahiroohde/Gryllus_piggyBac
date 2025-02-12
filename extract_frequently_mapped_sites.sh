###extract_frequently_mapped_sites.sh###

#!/bin/bash

# Directory containing BAM files
bam_dir="/path/to/bam"
output_dir="/path/to/mapped_sites"
mkdir -p "$output_dir"  # Ensure output directory exists

# Define proximity in base pairs for "closeness" of reads and frequency threshold
proximity=10000
frequency=10

# Loop through all BAM files in the directory
for bam_file in "$bam_dir"/*sorted.bam; do
    # Define output file for each BAM file
    output_file="${output_dir}/$(basename "$bam_file" .bam)_frequently_mapped_sites.txt"

    # Process each BAM file to count read alignments
    /usr/bin/samtools view -F 260 "$bam_file" | \
        awk -v prox="$proximity" -v freq="$frequency" '{
            # Extract reference scaffold and start position
            scaffold=$3; position=$4;
            # Create a key by dividing the position by the proximity threshold
            key=scaffold "_" int(position/prox);
            count[key]++;
        }
        END {
            # Print counts per proximity-bucketed position
            for (key in count)
                if (count[key] > freq)
                    print key, count[key];
        }' > "$output_file"

    # Optionally, display the results
    echo "Results for $(basename "$bam_file"):"
    cat "$output_file"
    echo "----"
done
