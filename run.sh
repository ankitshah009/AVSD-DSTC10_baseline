#!/bin/bash

. conda/bin/activate
conda activate avsd

datapath=data/features
#run='srun -p gpu --gres=gpu:1'  # if slurm is available
run=

exp_name=avsd01
train_set=./data/train_set4DSTC10-AVSD.json
val_set=./data/valid_set4DSTC10-AVSD+reason.json

# check if the log directory exists
if [ -d "log/${exp_name}/train_cap" ]; then
   echo \"./log/${exp_name}/train_cap\" already exists. Set a new exp_name different from \"${exp_name}\", or remove the directory
   exit
fi
# convert data
echo coverting json files to csv for the tool
python utils/generate_csv.py duration_info/duration_Charades_v1_480.csv $train_set train ./data/dstc10_train.csv
python utils/generate_csv.py duration_info/duration_Charades_v1_480.csv $val_set val ./data/dstc10_val.csv

# train
$run python main.py \
 --train_meta_path ./data/dstc10_train.csv \
 --val_meta_path ./data/dstc10_val.csv \
 --reference_paths $val_set \
 --video_features_path ${datapath}/video_feats/ \
 --audio_features_path ${datapath}/vggish/ \
 --procedure train_cap \
 --B 12 \
 --unfreeze_word_emb \
 --d_vid 2048 --d_aud 128 \
 --d_model_video 512 \
 --d_model_audio 64 \
 --d_model_caps 256 \
 --use_linear_embedder \
 --device_ids 0 \
 --one_by_one_starts_at 30 \
 --stopwords data/stopwords.txt \
 --exp_name $exp_name \
 --log_dir ./log
