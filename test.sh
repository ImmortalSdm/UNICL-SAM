device=$1
path=$2
epoch=$3

# caicl
for num_sample in 1 2 3;
do
    if [ $num_sample -eq 1 ]; then
        CUDA_VISIBLE_DEVICES=${device} python test.py \
        -cfg ${path}/config.yaml \
        -pt ${path}/checkpoint-${epoch}.pth \
        -d sam_seg \
        -s 518 \
        -b 4 \
        -val "test_sam_seg_sample-num-${num_sample}" \
        -n ${num_sample} \
        -id "configs/test/caicl_seg_3w_num_${num_sample}_id.json" \
        --uncertainty_vis True \
        --save_attn True \
        --cluster_vis True \
        
    else
        CUDA_VISIBLE_DEVICES=${device} python test.py \
        -cfg ${path}/config.yaml \
        -pt ${path}/checkpoint-${epoch}.pth \
        -d sam_seg \
        -s 518 \
        -b 4 \
        -val "test_sam_seg_sample-num-${num_sample}" \
        -n ${num_sample} \
        -id "configs/test/caicl_seg_3w_num_${num_sample}_id.json" \
        # --iou_pred True
    fi
done

# # coco-20i
for split in 0 1 2 3;
do
    CUDA_VISIBLE_DEVICES=${device} python test.py \
    -cfg ${path}/config.yaml \
    -pt ${path}/checkpoint-${epoch}.pth \
    -d fss \
    -s 518 \
    -b 4 \
    -val "test_coco_sample-num-1_split-${split}" \
    -n 1 \
    --split ${split}
done


# fss1000
for num_sample in 1;
do
    CUDA_VISIBLE_DEVICES=${device} python test.py \
        -cfg ${path}/config.yaml \
        -pt ${path}/checkpoint-${epoch}.pth \
        -d fss1000 \
        -s 518 \
        -b 4 \
        -val "test_fss1000_sample-num-${num_sample}" \
        -n ${num_sample}
done

# lvis-92i
for split in 0 1 2 3 4 5 6 7 8 9;
do
    CUDA_VISIBLE_DEVICES=${device} python test.py \
    -cfg ${path}/config.yaml \
    -pt ${path}/checkpoint-${epoch}.pth \
    -d lvis \
    -s 518 \
    -b 4 \
    -val "test_lvis_sample-num-1_split-${split}" \
    -n 1 \
    --split ${split}
done
