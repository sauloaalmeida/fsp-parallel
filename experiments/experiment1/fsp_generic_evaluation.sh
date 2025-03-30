#!/bin/bash

# defining input file
INPUT_FILE=$1

# check if the file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "File $INPUT_FILE not found!"
    exit 1
fi

# Loop to read each line in the file
while read -r idScenery databaseName distanceMethod numRepeats numFolds; do
    # Substitua "comando" pelo comando que deseja executar
    python3 -W ignore fsp_generic_evaluation.py $idScenery $databaseName $distanceMethod $numRepeats $numFolds > "output/output_$idScenery-$databaseName-$distanceMethod-$numRepeats-$numFolds.txt"
done < "$INPUT_FILE"