#!/bin/bash

# defining input file
INPUT_FILE=$1

# check if the file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "File $INPUT_FILE not found!"
    exit 1
fi

# Loop to read each line in the file
while read -r idScenery databaseName numRepeats numFolds numProcess; do
    python3 -W ignore fsp_parallel_evaluation.py $idScenery $databaseName $numRepeats $numFolds $numProcess > "output/output_$idScenery-$databaseName-$numProcess-$numRepeats-$numFolds.txt"
done < "$INPUT_FILE"