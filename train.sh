# ICL SAM sam_seg CUDA_VISIBLE_DEVICES=9 
CUDA_VISIBLE_DEVICES=9 python -u -m torch.distributed.run --master_port 25001 --nproc_per_node=1 train.py \
    --model vrp_sam_dino_uncertainty_graph_deterministic_contrastive \
    --input_size 518 \
    --samples_num 1 \
    --batch_size 4 \
    --accum_iter 2 \
    --warmup_epochs 4 \
    --epochs 20 \
    --save_ckpt_freq 2 \
    --output_dir experiments/cvpr_2025 \
    --data_type sam_seg_imgiter_degrad_contrastive \
    --config configs/unicl_sam/vrp_sam_dinov2_large_vitdet_fpn_uncertainty.yaml \
    # 2>&1 | tee vrp_sam_maskpool_dinov2_large_res518_uncertainty_rgcn_cluster10_cluster_loss_deterministic_avg_pool_uncertainty_graphcl_aug_random_reweighting_sa_icl_learn_sparse.log

# sh test.sh 9 experiments/cvpr_2025/00003-vrp_sam_dinov2_large_uncertainty_refine-vrp_sam_dino_uncertainty_graph_deterministic_contrastive-sam_seg_degrad_contrastive-epoch20-batch4-blr0.0001-res518-samples1 19
