#!/bin/bash

. conda/bin/activate avsd

if [ $# -lt 1 ];then
    echo "Evaluate generated answers and regions"
    echo "Usage: $0 submission.json [test_dstc7|test_dstc8|test_dstc7_multi|test_dstc8_multi]";
    exit 1
fi
reference=./data/valid_set4DSTC10-AVSD+reason.json
submission=$1
last_only=
if [ $# -eq 2 ]; then
    if [ $2 = "test_dstc8" ]; then
        # mock test with DSTC8 test set
        reference=./data/mock_test_set4DSTC10-AVSD_from_DSTC8_singref.json
    elif [ $2 = "test_dstc7" ]; then
        # mock test with DSTC7 test set
        reference=./data/mock_test_set4DSTC10-AVSD_from_DSTC7_singref.json
    elif [ $2 = "test_dstc8_multi" ]; then
        # mock test with DSTC8 test set with multiple references
        reference=./data/mock_test_set4DSTC10-AVSD_from_DSTC8_multiref.json
    elif [ $2 = "test_dstc7_multi" ]; then
        # mock test with DSTC7 test set with multiple references
        reference=./data/mock_test_set4DSTC10-AVSD_from_DSTC7_multiref.json
    else
        echo "Error: unknown testset specifier: $2"
        echo "The testset should be one of [test_dstc7, test_dstc8, test_dstc7_multi, test_dstc8_multi]"
        exit 1
    fi
    last_only=--last_only  # evaluate only the last answer in each dialog
fi

echo Reference: $reference
echo Submission: $submission
python evaluation/evaluate.py \
        -S ./data/stopwords.txt \
        -r $reference \
        -s $submission \
        $last_only
