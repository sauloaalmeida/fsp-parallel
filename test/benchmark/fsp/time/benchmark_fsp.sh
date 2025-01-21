#!/bin/bash
date
cat $1 | while read line || [[ -n $line ]];
do
   # do something with $line here
   python3 -W ignore fsp_time_evaluation.py $line
done
date
