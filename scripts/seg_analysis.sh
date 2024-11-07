#!/bin/bash
device=$1
path=$2
epoch=$3

for degrad_type in "gray" "sobel" "canny" "bbox" "binary" "dilate" "erode" "horizontal_flip" "vertical_flip" "equalize" "cartoon";
do
    for num_sample in 1 2 3;
    do
        CUDA_VISIBLE_DEVICES=${device} python scripts/seg_analysis.py \
        -cfg ${path}/config.yaml \
        -pt ${path}/checkpoint-${epoch}.pth \
        -d degrad \
        -s 512 \
        -b 4 \
        -val "test_sam_seg_sample_${degrad_type}_refer-num-${num_sample}" \
        -n ${num_sample} \
        -t ${degrad_type}
    done
done

# gaussian
for num_sample in 1 2 3;
do
    for kernel in 7 11 15;
    do
        CUDA_VISIBLE_DEVICES=${device} python scripts/seg_analysis.py \
        -cfg ${path}/config.yaml \
        -pt ${path}/checkpoint-${epoch}.pth \
        -d degrad \
        -s 512 \
        -b 4 \
        -val "test_sam_seg_sample_gaussian_k${kernel}_refer-num-${num_sample}" \
        -n ${num_sample} \
        -t gaussian \
        --kernel ${kernel}
    done
done

# color_jitter
for num_sample in 1 2 3;
do
    for bright in 0.5 1.5;
    do
        for contrast in 0.5 1.5;
        do
            for saturation in 0.5 1.5;
            do
                for hue in 0.1 0.3 0.5;
                do
                    CUDA_VISIBLE_DEVICES=${device} python scripts/seg_analysis.py \
                    -cfg ${path}/config.yaml \
                    -pt ${path}/checkpoint-${epoch}.pth \
                    -d degrad \
                    -s 512 \
                    -b 4 \
                    -val "test_sam_seg_sample_color_jitter_b${bright}_c${contrast}_s${saturation}_h${hue}_refer-num-${num_sample}" \
                    -n ${num_sample} \
                    -t color_jitter \
                    --bright ${bright} \
                    --contrast ${contrast} \
                    --saturation ${saturation} \
                    --hue ${hue}
                done
            done
        done
    done
done

# sharp
for num_sample in 1 2 3;
do
    for sharp in 5 10 15;
    do
        CUDA_VISIBLE_DEVICES=${device} python scripts/seg_analysis.py \
        -cfg ${path}/config.yaml \
        -pt ${path}/checkpoint-${epoch}.pth \
        -d degrad \
        -s 512 \
        -b 4 \
        -val "test_sam_seg_sample_sharp_s${sharp}_refer-num-${num_sample}" \
        -n ${num_sample} \
        -t sharp \
        --sharp ${sharp}
    done
done

# posterize
for num_sample in 1 2 3;
do
    for bit in 1 2 3;
    do
        CUDA_VISIBLE_DEVICES=${device} python scripts/seg_analysis.py \
        -cfg ${path}/config.yaml \
        -pt ${path}/checkpoint-${epoch}.pth \
        -d degrad \
        -s 512 \
        -b 4 \
        -val "test_sam_seg_sample_posterize_bit${bit}_refer-num-${num_sample}" \
        -n ${num_sample} \
        -t posterize \
        --bit ${bit}
    done
done

# solarize
for num_sample in 1 2 3;
do
    for threshold in 0 64 128 192 256;
    do
        CUDA_VISIBLE_DEVICES=${device} python scripts/seg_analysis.py \
        -cfg ${path}/config.yaml \
        -pt ${path}/checkpoint-${epoch}.pth \
        -d degrad \
        -s 512 \
        -b 4 \
        -val "test_sam_seg_sample_solarize_t${threshold}_refer-num-${num_sample}" \
        -n ${num_sample} \
        -t solarize \
        --threshold ${threshold}
    done
done

# jpeg_compression
for num_sample in 1 2 3;
do
    for jpeg in 5 10 20;
    do
        CUDA_VISIBLE_DEVICES=${device} python scripts/seg_analysis.py \
        -cfg ${path}/config.yaml \
        -pt ${path}/checkpoint-${epoch}.pth \
        -d degrad \
        -s 512 \
        -b 4 \
        -val "test_sam_seg_sample_jpeg_compression_ratio${jpeg}_refer-num-${num_sample}" \
        -n ${num_sample} \
        -t jpeg_compression \
        --jpeg ${jpeg}
    done
done

# gaussian_noise
for num_sample in 1 2 3;
do
    for var in 5000 50000 500000;
    do
        CUDA_VISIBLE_DEVICES=${device} python scripts/seg_analysis.py \
        -cfg ${path}/config.yaml \
        -pt ${path}/checkpoint-${epoch}.pth \
        -d degrad \
        -s 512 \
        -b 4 \
        -val "test_sam_seg_sample_gaussian_noise_m0_var${var}_refer-num-${num_sample}" \
        -n ${num_sample} \
        -t gaussian_noise \
        --var ${var}
    done
done

# motion_blur
for num_sample in 1 2 3;
do
    for blur in 5 9 15;
    do
        CUDA_VISIBLE_DEVICES=${device} python scripts/seg_analysis.py \
        -cfg ${path}/config.yaml \
        -pt ${path}/checkpoint-${epoch}.pth \
        -d degrad \
        -s 512 \
        -b 4 \
        -val "test_sam_seg_sample_motion_blur_b${blur}_refer-num-${num_sample}" \
        -n ${num_sample} \
        -t motion_blur \
        --blur ${blur}
    done
done

# mean_shift_blur
for num_sample in 1 2 3;
do
    for color in 1 10 50;
    do
        CUDA_VISIBLE_DEVICES=${device} python scripts/seg_analysis.py \
        -cfg ${path}/config.yaml \
        -pt ${path}/checkpoint-${epoch}.pth \
        -d degrad \
        -s 512 \
        -b 4 \
        -val "test_sam_seg_sample_mean_shift_blur_c${color}_refer-num-${num_sample}" \
        -n ${num_sample} \
        -t mean_shift_blur \
        --color ${color}
    done
done

# light
for num_sample in 1 2 3;
do
    for light in 0.3 0.7 1.3 1.7;
    do
        CUDA_VISIBLE_DEVICES=${device} python scripts/seg_analysis.py \
        -cfg ${path}/config.yaml \
        -pt ${path}/checkpoint-${epoch}.pth \
        -d degrad \
        -s 512 \
        -b 4 \
        -val "test_sam_seg_sample_light_l${light}_refer-num-${num_sample}" \
        -n ${num_sample} \
        -t light \
        --light ${light}
    done
done