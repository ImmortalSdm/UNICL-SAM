# ICL SAM sam_seg CUDA_VISIBLE_DEVICES=9 
CUDA_VISIBLE_DEVICES=8,9 python -u -m torch.distributed.run --master_port 25002 --nproc_per_node=2 train.py \
    --model vrp_sam_dino_uncertainty_graph_deterministic_contrastive_inst \
    --input_size 518 \
    --samples_num 1 \
    --batch_size 2 \
    --accum_iter 4 \
    --warmup_epochs 4 \
    --epochs 20 \
    --save_ckpt_freq 2 \
    --output_dir experiments/cvpr_2025 \
    --data_path /home/dmsheng/datasets/image_inpainting/ILSVRC_2012/ \
    --data_type inst \
    --config configs/unicl_sam/vrp_sam_dinov2_large_vitdet_fpn_uncertainty.yaml \
    2>&1 | tee cvpr_coco-ade-lvis-sem_coco-lvis-inst_fix-dense.log

# sh test.sh 9 experiments/cvpr_2025/00003-vrp_sam_dinov2_large_uncertainty_refine-vrp_sam_dino_uncertainty_graph_deterministic_contrastive-sam_seg_degrad_contrastive-epoch20-batch4-blr0.0001-res518-samples1 19
