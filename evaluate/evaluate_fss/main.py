from glob import glob
from pathlib import Path
import shutil
import torch
import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np
from torchmetrics.classification import BinaryJaccardIndex, JaccardIndex

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str)
    parser.add_argument('--datasets', type=str)
    parser.add_argument('--path', type=str)
    parser.add_argument('--type', type=str)
    parser.add_argument('--epoch', type=str)
    parser.add_argument('--gt_root_dir', type=str)
    parser.add_argument('--pred_root_dir', type=str)
    parser.add_argument('--save_dir', type=str, default='./score/')
    parser.add_argument('--cuda', type=bool, default=True)
    config = parser.parse_args()

    return config

if __name__ == "__main__":
    config = get_args()

    if config.type == 'caicl':
        num = 3
        offset = 1
        name = 'test_sam_seg_sample-num'
    elif config.type == 'fss1000':
        num = 1
        offset = 1
        name = 'test_fss1000_sample-num'        
    elif config.type == 'lvis':
        num = 10
        offset = 0
        name = 'test_lvis_sample-num-1_split'    
    else:
        num = 4
        offset = 0
        name = 'test_coco_sample-num-1_split'

    shutil.rmtree('./temp/pred_maps/')
    shutil.rmtree('./temp/gt/')

    for i in range(num):
        data_path = f'{config.path}/checkpoint-{config.epoch}.pth/{name}-{i+offset}'
        gt_paths = sorted(glob(rf"{data_path}/gt*.png"))
        r_paths = sorted(glob(rf"{data_path}/r*.png"))

        gt_save_path = f'./temp/gt/coco-512-{i+1}' # seggpt-q
        exp_name = config.path.split('/')[-1]
        r_save_path = f'./temp/pred_maps/{exp_name}/coco-512-{i+1}'

        Path(gt_save_path).mkdir(parents=True, exist_ok=True)
        Path(r_save_path).mkdir(parents=True, exist_ok=True)
        
        gt_masks = []
        q_gt_masks = []
        re_masks = []

        connected = []

        length = len(gt_paths)
        for j in tqdm(range(length)):
            gt_path = gt_paths[j]
            r_path = r_paths[j]

            gt_canvas = cv2.imread(gt_path)
            result = cv2.imread(r_path)

            H, W, C = gt_canvas.shape

            gt = gt_canvas[int(H-512):, int(W-512):, :]
            result = result[int(H-512):, int(W-512):, :]

            t, gt = cv2.threshold(gt, 127, 1, cv2.THRESH_BINARY)
            t, result = cv2.threshold(result, 127, 1, cv2.THRESH_BINARY)
            gt_masks.append(gt[..., 0])
            re_masks.append(result[..., 0])

            n_labels, _, _, _ = cv2.connectedComponentsWithStats(result[..., 0], 8)
            connected.append(n_labels)

            cv2.imwrite(os.path.join(gt_save_path, str(j)+'.png'), gt[..., 0]*255)
            cv2.imwrite(os.path.join(r_save_path, str(j)+'.png'), result[..., 0]*255)

        seg_gt_masks = np.stack(gt_masks, axis=0)
        seg_gt_masks = torch.from_numpy(seg_gt_masks).to(torch.int8)
        seg_re_masks = np.stack(re_masks, axis=0)
        seg_re_masks = torch.from_numpy(seg_re_masks).to(torch.int8)

        # import pdb; pdb.set_trace()

        # IoUs = []
        # for i in range(len(seg_re_masks)):
        #     IoUs.append(bjaccard(torch.from_numpy(seg_re_masks[i]), torch.from_numpy(seg_gt_masks[i])))
        
        # import pickle
        # pickle.dump(torch.stack(IoUs), file=open('/home/qchugroup/sdmcvpr2025/code/visual_prompting/analysis/iou/dinov2_large_deepcut_graph_cluster10_kmeans_ce_loss_miou.pkl', 'wb+'))

        # import pdb; pdb.set_trace()

        jaccard_b_ig = JaccardIndex(task="binary", num_classes=2, ignore_index=0)
        bjaccard = BinaryJaccardIndex()

        r_MIoU = jaccard_b_ig(seg_re_masks, seg_gt_masks) # 0.4936
        br_MIoU = bjaccard(seg_re_masks, seg_gt_masks) # 0.4936
        connected = torch.tensor(connected).float().mean()
        mae = torch.abs(seg_re_masks - seg_gt_masks).float().mean() 

        print('R_MIoU: ', round(r_MIoU.item()*100, 2))
        print('BR_MIoU: ', round(br_MIoU.item()*100, 2))
        print('MAE: ', round(mae.item(), 3))
        print('connected: ', round(connected.item(), 2))

