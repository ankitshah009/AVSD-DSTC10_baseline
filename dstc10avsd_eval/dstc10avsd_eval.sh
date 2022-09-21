#!/bin/bash

if [ $# -lt 1 ];then
    echo "Evaluate generated answers and regions for DSTC10-AVSD"
    echo "Usage: $0 submission.json";
    exit 1
fi
reference=./data/test_set4DSTC10-AVSD_multiref+reason.json
stopwords=./data/stopwords.txt
submission=$1

echo Reference: $reference
echo Submission: $submission
python3 evaluation/evaluate.py \
        -S $stopwords \
        -r $reference \
        -s "$submission" \
        --last_only
