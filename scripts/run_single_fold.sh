#!/bin/bash

set -e

cd ../

if [ -z "$1" ]
    then
    echo "No argument supplied"
    exit 1
fi

exp_name=cross_val_test
dataset=cross_val
iFold=$1
train_set_name=split/${dataset}/${iFold}_train.csv
val_set_name=split/${dataset}/${iFold}_val.csv
test_set_name=$val_set_name
mask_out_dir=results/${exp_name}/${iFold}_mask
rcnn_out_dir=results/${exp_name}/${iFold}_rcnn
rpn_out_dir=results/${exp_name}/${iFold}_rpn
mask_ckpt_path=${mask_out_dir}/model/200.ckpt
rcnn_ckpt_path=${rcnn_out_dir}/model/200.ckpt
rpn_ckpt_path=${rpn_out_dir}/model/200.ckpt

# # Training with mask
python train.py --train-set-list $train_set_name --val-set-list $val_set_name --out-dir $mask_out_dir --epoch-mask 80 --epoch-rcnn 65
python test_LIDC.py eval --test-set-name $test_set_name --weight $mask_ckpt_path --out-dir $mask_out_dir

# # Training without mask, no need to train from the start
python train.py --train-set-list $train_set_name --val-set-list $val_set_name --out-dir $rcnn_out_dir --epoch-mask 400 --epoch-rcnn 65 --ckpt ${mask_out_dir}/model/079.ckpt
python test_LIDC.py eval --test-set-name $test_set_name --weight $rcnn_ckpt_path --out-dir $rcnn_out_dir

# # Training without mask and rcnn, no need to train from the start
python train.py --train-set-list $train_set_name --val-set-list $val_set_name --out-dir $rpn_out_dir --epoch-mask 400 --epoch-rcnn 300 --ckpt ${mask_out_dir}/model/064.ckpt
python test_LIDC.py eval --test-set-name $test_set_name --weight $rpn_ckpt_path --out-dir $rpn_out_dir
