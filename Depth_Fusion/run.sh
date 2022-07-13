#!/usr/bin/env bash

root_path="../small_dtu_results"
save_path="../small_dtu_points"
test_list="./dataloader/datalist/small_dtu/test.txt"

python ./Depth_Fusion/fuse.py --root_path $root_path --save_path $save_path \
                --data_list $test_list --dist_thresh 0.001 --prob_thresh 0.6 \
                --num_consist 10 --device "cuda"