from glob import glob
import json
from pathlib import Path
import torch
import torch.nn as nn
import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np
from torchmetrics.classification import BinaryJaccardIndex, JaccardIndex
import shutil
import re
from terminaltables import AsciiTable

degrad_lists = ['gaussian', 'gray', 'color_jitter', 'sharp', 'horizontal_flip', 'vertical_flip', 
                'posterize', 'solarize', 'equalize', 'jpeg_compression', 'gaussian_noise', 
                'motion_blur', 'cartoon', 'mean_shift_blur', 'light', 'sobel', 'canny',
                'binary', 'bbox', 'dilate', 'erode']

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
    parser.add_argument('--save_json', type=bool, default=True)
    
    config = parser.parse_args()

    return config

if __name__ == "__main__":
    config = get_args()
    # ann_path = '/home/dmsheng/datasets/coco/annotations/val_gtav_seg_3w.json'
    ann_path = '/home/dmsheng/datasets/coco/annotations/val_sd_context_seg_3w.json'
    with open(ann_path,'r') as fr: 
        sd_json = json.load(fr) 

    with open('/home/dmsheng/datasets/coco/annotations/val_seg_3w.json','r') as fr: 
        coco_json = json.load(fr) 

    cats = [int(k) for k,v in sd_json['per_cat_pool'].items() if len(v) > 0]
    img_ids = [int(k) for k,v in coco_json['dataset_dicts'].items() if v['category_id'] in cats]

    try:
        shutil.rmtree('evaluate/evaluate_seg/temp/pred_maps/')
        shutil.rmtree('evaluate/evaluate_seg/temp/gt/')
    except:
        pass

    pred_path = f'{config.path}/checkpoint-{config.epoch}.pth/'
    degrad_score = {}

    jaccard_b_ig = JaccardIndex(task="binary", num_classes=2, ignore_index=0)
    bjaccard = BinaryJaccardIndex()

    print(f'\nStarting degard evaluation!\n')
    for degrad in degrad_lists:
        degrad_score[degrad] = {
            'exp': {}
        }
        degrad_paths = sorted(glob(f'{pred_path}*{degrad}*'))

        results_per_degrad = []

        for num in range(3):

            name = f'num-{num+1}'
            exp = []
            match_degrad_paths = set(sorted([s for s in degrad_paths if re.search(f'num-{num+1}', s)]))

            results_per_degrad_per_num = []
            if len(match_degrad_paths) > 0:
                for data_path in match_degrad_paths:
                    gt_paths = sorted(glob(rf"{data_path}/gt*.png"))
                    exp_name = data_path.split('/')[-1]

                    if len(gt_paths) == 0:
                        print(f'Empty paths for {degrad} {exp_name}!\n')
                        continue
                    r_paths = sorted(glob(rf"{data_path}/r*.png"))


                    gt_save_path = f'./temp/gt/coco-512-{exp_name}' # seggpt-q
                    r_save_path = f'./temp/pred_maps/painter/coco-512-{exp_name}'

                    Path(gt_save_path).mkdir(parents=True, exist_ok=True)
                    Path(r_save_path).mkdir(parents=True, exist_ok=True)
                    
                    gt_masks = []
                    q_gt_masks = []
                    re_masks = []

                    connected = []

                    length = len(gt_paths)

                    for j in range(length):
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

                    connected = torch.tensor(connected).float().mean()

                    r_MIoU = jaccard_b_ig(seg_re_masks, seg_gt_masks) # 0.4936
                    br_MIoU = bjaccard(seg_re_masks, seg_gt_masks) # 0.4936
                    mae = torch.abs(seg_re_masks - seg_gt_masks).float().mean() 

                    result = {
                        'R_MIoU': round(r_MIoU.item()*100, 4),
                        'BR_MIoU': round(br_MIoU.item()*100, 4),
                        'connected': round(connected.item(), 4),
                        'MAE': round(mae.item(), 4),
                    }
                    exp.append(result)

                    res_item = [f'{exp_name}', f'{float(r_MIoU.item()*100):0.4f}', f'{float(br_MIoU.item()*100):0.4f}', f'{float(mae.item()):0.4f}', f'{float(connected.item()):0.4f}']
                    results_per_degrad_per_num.append(res_item)
            else:
                print(f'\nEmpty paths for {degrad}-{name}!\n')
                continue

            if len(exp) > 0:
                global_mean_R_MIoU = np.mean(np.array([e['R_MIoU'] for e in exp]))
                global_mean_BR_MIoU = np.mean(np.array([e['BR_MIoU'] for e in exp]))
                global_mean_connected = np.mean(np.array([e['connected'] for e in exp]))
                global_mean_MAE = np.mean(np.array([e['MAE'] for e in exp]))
            else:
                global_mean_R_MIoU = global_mean_BR_MIoU = global_mean_connected = global_mean_MAE = 0.

            degrad_score[degrad]['exp'][name] = {
                'exp': exp,
                'R_MIoU': round(global_mean_R_MIoU, 2),
                'BR_MIoU': round(global_mean_BR_MIoU, 2),
                'connected': round(global_mean_connected, 2),
                'MAE': round(global_mean_MAE, 3),
            }

            res_item = [f'Mean results', 
                        f'{float(global_mean_R_MIoU):0.4f}', 
                        f'{float(global_mean_BR_MIoU):0.4f}', 
                        f'{float(global_mean_MAE):0.4f}', 
                        f'{float(global_mean_connected):0.4f}']
            results_per_degrad_per_num.append(res_item)            

            headers = [f'{degrad}_{name}', 'R_MIoU', 'BR_MIoU', 'MAE', 'connected']
            table_data = [headers] + results_per_degrad_per_num
            table = AsciiTable(table_data)
            print(f'\nResults for degard {degrad} refer num {num+1}:\n' + table.table)
            res_item = [f'{degrad}_{name}', 
                        f'{float(global_mean_R_MIoU):0.2f}', 
                        f'{float(global_mean_BR_MIoU):0.2f}', 
                        f'{float(global_mean_MAE):0.3f}', 
                        f'{float(global_mean_connected):0.2f}']
            results_per_degrad.append(res_item)

        degrad_score[degrad]['R_MIoU'] = round(np.mean(np.array([v['R_MIoU'] for k,v in degrad_score[degrad]['exp'].items()])), 2)
        degrad_score[degrad]['BR_MIoU'] = round(np.mean(np.array([v['BR_MIoU'] for k,v in degrad_score[degrad]['exp'].items()])), 2)
        degrad_score[degrad]['connected'] = round(np.mean(np.array([v['connected'] for k,v in degrad_score[degrad]['exp'].items()])), 2)
        degrad_score[degrad]['MAE'] = round(np.mean(np.array([v['MAE'] for k,v in degrad_score[degrad]['exp'].items()])), 2)
    
        headers = [f'Results for degrad {degrad}', 'R_MIoU', 'BR_MIoU', 'MAE', 'connected']
        table_data = [headers] + results_per_degrad

        table = AsciiTable(table_data)
        print(f'\nResults for degard {degrad}:\n' + table.table)        

    if config.save_json:
        exp = config.path.split('/')[-1]
        save_name = f'logs/{exp}-checkpoint-{config.epoch}-degrad.json'
        with open(save_name, 'w') as f: 
            json.dump(degrad_score, f)    

    print(f'\nFinish degard evaluation!\n')

