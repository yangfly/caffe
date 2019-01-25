#!/bin/bash

redo=false
dataset_name="ILSVRC2016"
data_root_dir="data/$dataset_name"
mapfile="$data_root_dir/labelmap_ilsvrc_det.prototxt"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if $redo
then
  extra_cmd="$extra_cmd --redo"
fi

for dataset in test
do
  python tools/create_annoset.py --anno-type="classification" --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $data_root_dir/$dataset".txt" $data_root_dir/$db/DET/$dataset_name"_"$dataset"_"$db 2>&1 | tee $data_root_dir/$dataset.log
done

for dataset in val2 trainval1
do
  python tools/create_annoset.py --anno-type="detection" --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $data_root_dir/$dataset".txt" $data_root_dir/$db/DET/$dataset_name"_"$dataset"_"$db 2>&1 | tee $data_root_dir/$dataset.log
done
