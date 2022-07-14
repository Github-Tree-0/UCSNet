#!/usr/bin/env bash

test_list="./dataloader/datalist/small_dtu/test.txt"
root_path="../Small_DTU"
save_path="../small_dtu_results"
points_save_path="../small_dtu_points"

CUDA_VISIBLE_DEVICES=0 python test.py --root_path $root_path --test_list $test_list --save_path $save_path \
                --max_h 1200 --max_w 1600

CUDA_VISIBLE_DEVICES=0 python ./Depth_Fusion/fuse.py --root_path $save_path --save_path $points_save_path \
                --data_list $test_list --dist_thresh 0.001 --prob_thresh 0.6 \
                --num_consist 2 --device "cuda"