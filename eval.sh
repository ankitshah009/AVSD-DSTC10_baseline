#!/bin/bash

. conda/bin/activate
conda activate avsd

if [ $# -lt 1 ];then
    echo "Evaluate generated answers and regions"
    echo "Usage: $0 submission.json [test]";
    exit 1
fi
reference=./data/valid_set4DSTC10-AVSD+reason.json
submission=$1
last_only=
if [ $# -eq 2 ] && [ $2 = "test" ]; then
    # mock test with DSTC8 test set
    reference=./data/mock_test_set4DSTC10-AVSD_from_DSTC8_singref.json
    last_only=--last_only  # evaluate only the last answer in each dialog
fi

echo Reference: $reference
echo Submission: $submission
python evaluation/evaluate.py \
        -S ./data/stopwords.txt \
        -r $reference \
        -s $submission \
        $last_only
