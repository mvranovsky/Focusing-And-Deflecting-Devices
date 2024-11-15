#!/bin/bash
#---------------------------------------------
# run by command:
# ./RunParallel.sh Run1 10 ../../MAXIMA/analyticalResultsP.txt specialAssignment.py specialAssignment.zip
#
# Run1 = basedir name
# 10 = number of independent executions
# inputlist
# script to execute
# all files needed for running including the script for execution (absolute path)
#---------------------------------------------
# Check if the correct number of arguments are provided

# Function to split lines into groups and distribute them to .txt files in directories
split_lines_into_groups() {
    local input_file="$1"
    local num_groups="$2"
    local base_dir="$3"
    
    # Read the total number of lines in the input file
    total_lines=$(wc -l < "$input_file")
    
    # Calculate the number of lines per group
    lines_per_group=$(( (total_lines + num_groups - 1) / num_groups ))  # Round up
    
    # Initialize variables
    group_num=1
    line_counter=0
    
    # Create a file for each group and add lines to it
    while IFS= read -r line; do
        # Create a directory for the group if it doesn't exist
        group_dir="$base_dir/dir$group_num"
        
        # Write the current line to the appropriate group's file
        echo "$line" >> "$group_dir/inputData$group_num.txt"
        
        # Move to the next group if the number of lines in the current group reaches the limit
        ((line_counter++))
        if [ "$line_counter" -ge "$lines_per_group" ]; then
            ((group_num++))
            line_counter=0
        fi
        
        # If we've assigned all groups, restart at group 1 (cyclic distribution)
        if [ "$group_num" -gt "$num_groups" ]; then
            group_num=1
        fi
    done < "$input_file"
}



moveFiles(){

	#local filesToBeMoved=("specialAssignment.py" "parallelBeam.in" "aperture1.dat" "aperture2.dat" "aperture3.dat")
	#filesToBeMoved+=("Astra")
	#filesToBeMoved+=("generator")
	#filesToBeMoved+=("test.ini")
	#filesToBeMoved+=("test0.ini")
	#filesToBeMoved+=("test1.ini")
	#filesToBeMoved+=("test2.ini")
	#filesToBeMoved+=("test3.ini")
	#filesToBeMoved+=("test4.ini")

	for (( j = 1; j <= NUM_EXECUTIONS; j++ )); do
		mkdir -p "$BASE_DIR/dir$j"
        cp "$ZIP_FILE" "$BASE_DIR/dir$j"
        cd "$BASE_DIR/dir$j"
        unzip "$ZIP_FILE"
        touch "output.txt"
        cd "../../"
	#	for file in "${filesToBeMoved[@]}"; do
	#		cp "../specialAssignment/$file" "$BASE_DIR/dir$j"
	#	done
	#	touch "$BASE_DIR/dir$j/output.txt" 
	#	touch "$BASE_DIR/dir$j/errors.txt" 
	#	mkdir "$BASE_DIR/dir$j/resFigs"
	#	touch "$BASE_DIR/dir$j/inputData$j.txt" 
	done


	echo "Succesfully created directories for each execution and moved files inside."
}

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <base_directory> <number_of_parallel_executions> <input_file> <executable_script> <zip_file>"
    exit 1
fi


# Read arguments
BASE_DIR=$1
NUM_EXECUTIONS=$2
INPUT_FILE=$3
SCRIPT=$4
ZIP_FILE=$5


# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE does not exist."
    exit 1
fi

# Check if the input file exists
if [ ! -f "$ZIP_FILE" ]; then
    echo "Error: zip file $ZIP_FILE does not exist."
    exit 1
fi


echo "Beginning submitting parallel processes."
echo "Base directory:                      $BASE_DIR"
echo "Number of parallel executions:       $NUM_EXECUTIONS"
echo "Running input file:                  $INPUT_FILE"
echo "Running script:                      $SCRIPT"
echo "All files are included in:           $ZIP_FILE"


# Create the base directory if it does not exist
mkdir -p "$BASE_DIR"

# Ensure a clean start
rm -rf "$BASE_DIR/*"

moveFiles

# Array to store the list of arguments for the Python script
args=()

# Read the input file into an array
while IFS= read -r line; do
    args+=("$line")
    #echo "$line"
done < "$INPUT_FILE"

# Check if we have enough arguments for the number of executions
if [ "${#args[@]}" -lt "$NUM_EXECUTIONS" ]; then
    echo "Error: Not enough arguments in $INPUT_FILE for the specified number of executions."
    echo "Number of inputs from list: ${#args[@]}"
    exit 1
fi

#split the lines to groups and write them to separate directories
split_lines_into_groups "$INPUT_FILE" "$NUM_EXECUTIONS" "$BASE_DIR"

# Create directories and execute the Python script
for ((i=1; i<=NUM_EXECUTIONS; i++)); do
    DIR="$BASE_DIR/dir$i"

    # Prepare the argument for the Python script
    ARG="${args[$((i-1))]}"
    
    # Run the Python script in the directory with the argument
    (
        cd "$DIR" || exit
        echo "Running script in $DIR"
        python3 "$SCRIPT" "inputData$i.txt"
        #echo "0.1 2 3 80" >> "results.txt"
        #echo "0.2 5 6 90" >> "results.txt"
        #cp "/home/michal/Desktop/RPIT/ASTRA/parallelFocusing/table.csv" "."
    ) &
done

# Wait for all background processes to finish
wait

echo "All executions are complete. Now moving on to finish up."

#python3 MergeAndSort.py "$BASE_DIR"

wait

echo "Data merged and saved to $BASE_DIR. Leaving."