#!/bin/bash

. conda/bin/activate avsd

datapath=data/features
#run='srun -p gpu --gres=gpu:1'  # if slurm is available
run=

exp_name=avsd01
# use validation set instead of test set before the test set gets available
test_set=./data/valid_set4DSTC10-AVSD+reason.json
test_csv=./data/dstc10_val.csv
featpath_suffix=
last_only=
log_dir=./log

if [ $# -eq 1 ]; then
  if [ $1 = "test_dstc8" ]; then
    # mock test with DSTC8 test set
    test_set=./data/mock_test_set4DSTC10-AVSD_from_DSTC8_singref.json
  elif [ $1 = "test_dstc7" ]; then
    # mock test with DSTC7 test set
    test_set=./data/mock_test_set4DSTC10-AVSD_from_DSTC7_singref.json
  elif [ $1 = "test_dstc8_multi" ]; then
    # mock test with DSTC8 test set with multiple references
    test_set=./data/mock_test_set4DSTC10-AVSD_from_DSTC8_multiref.json
  elif [ $1 = "test_dstc7_multi" ]; then
    # mock test with DSTC7 test set with multiple references
    test_set=./data/mock_test_set4DSTC10-AVSD_from_DSTC7_multiref.json
  elif [ $1 = "test" ]; then
    # official test set (not provided at the moment)
    test_set=./data/test_set4DSTC10-AVSD.json
  else
    echo "Error: unknown testset specifier: $2"
    echo "The testset should be one of [test_dstc7, test_dstc8, test_dstc7_multi, test_dstc8_multi, test]"
    exit 1
  fi
  test_csv=./data/dstc10_test.csv
  featpath_suffix=_testset
  last_only=--last_only  # evaluate only the last answer of each dialog
  python utils/generate_csv.py duration_info/duration_Charades_vu17_test_480.csv $test_set test $test_csv
else
  echo Run \"run_gen.sh test\" if official testset \"./data/test_set4DSTC10-AVSD.json\" is available
fi

# eval
echo Answer generation and evaluation for $test_set
$run python main.py \
 --train_meta_path ./data/dstc10_train.csv \
 --test_meta_path $test_csv \
 --reference_paths $test_set \
 --video_features_path ${datapath}/video_feats$featpath_suffix/ \
 --audio_features_path ${datapath}/vggish$featpath_suffix/ \
 --procedure eval_cap \
 --pretrained_cap_model_path ${log_dir}/${exp_name}/train_cap/best_cap_model.pt \
 --B 12 \
 --stopwords data/stopwords.txt \
 --exp_name $exp_name \
 --log_dir $log_dir \
 $last_only
