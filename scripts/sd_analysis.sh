device=$1
path=$2
epoch=$3

for num_sample in 1 2 3;
do
    CUDA_VISIBLE_DEVICES=${device} python scripts/seg_analysis.py \
    -cfg ${path}/config.yaml \
    -pt ${path}/checkpoint-${epoch}.pth \
    -d scale \
    -s 512 \
    -b 4 \
    -val "test_sam_seg_sample_scale1.0_refer_sd_xpaste_aug-num-${num_sample}" \
    -n ${num_sample} \
    -t 1.0
done
