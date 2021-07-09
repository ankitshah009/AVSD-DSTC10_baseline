#!/bin/bash

. conda/bin/activate
conda activate avsd

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

if [ $# -eq 1 ] && [ $1 = "test" ];then
  # mock test with DSTC8 test set
  test_set=./data/mock_test_set4DSTC10-AVSD_from_DSTC8_singref.json
  test_csv=./data/dstc10_test.csv
  featpath_suffix=_testset
  last_only=--last_only  # evaluate only the last answer of each dialog
  echo Coverting json files to csv for the system
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

