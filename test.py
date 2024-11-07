import argparse
from glob import glob
import json
from math import sqrt
import os
import random
from PIL import Image
import numpy as np
import cv2
import torch
from pathlib import Path

from torchvision import datasets
from torchmetrics.classification import BinaryJaccardIndex
from omegaconf import OmegaConf
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import torch.nn as nn

from unicl_sam.data import *
from utils.misc import set_random_seed
from utils.args import model_lists, data_lists

from unicl_sam.model.model import ICL_VRP_SAM_DINO_VitDet_FPN_Uncertatinty_Deterministic_Contrastive
import torch.nn.functional as F

imagenet_mean = torch.tensor([0.5, 0.5, 0.5])
imagenet_std = torch.tensor([0.5, 0.5, 0.5])

def normal(input):
    min = torch.min(input)
    input -= min
    max = torch.max(input)
    input /= max
    return input

'''
cv
'''

@torch.no_grad()
def generate_seg_class_img(model, data, args, config, device, num):
    refers = data['refer']
    refer_img, cat, gt_mask, gt_id = refers['input'], refers['cat'], refers['output'], refers['img_id']

    refer_img = refer_img.to(device, non_blocking=True)
    refer_img_tokens = model.tokenizer.img_tokenizer.get_codebook_indices(refer_img).flatten(1)
    B, N = refer_img_tokens.shape
    refer_img_emb = model.tokenizer.img_tokenizer.quantize.get_codebook_entry(refer_img_tokens, (B,int(sqrt(N)),int(sqrt(N)),256))
    q_refer_img = model.tokenizer.img_tokenizer.decode(refer_img_emb).cpu()
    q_refer_img = torch.clip(q_refer_img.detach().cpu() *255, 0, 255)
    imgs = torch.clip(refer_img.detach().cpu() *255, 0, 255)
    masks = torch.clip(gt_mask.detach().cpu() *255, 0, 255)

    if args.model == 'gpt_moe':
        if config.tokenizer.use_map:
            special_token_ids = {'pad_token_id': model.tokenizer.txt_tokenizer.pad_token_id,
                                'eos_token_id': model.tokenizer.txt_tokenizer.eos_token_id}
            bad_words = torch.arange(51271) # torch.arange(51294, 52295) # torch.cat((torch.arange(50270), torch.arange(51294, 52295)))
            result_img = model.inference(data, args.data_type, special_token_ids=special_token_ids, bad_words=bad_words) # , sample=True
        else:
            special_token_ids = {'pad_token_id': model.tokenizer.txt_tokenizer.pad_token_id + model.img_token_num,
                                'eos_token_id': model.tokenizer.txt_tokenizer.eos_token_id + model.img_token_num}
            bad_words = torch.arange(1024, len(model.tokenizer))
            result_img = model.inference(data, args.data_type, special_token_ids=special_token_ids, bad_words=bad_words)
    else:
        result_img = model.generate_img(data, args.data_type, max_length=256) # 256    

    result_img[result_img<=128] = 0
    result_img[result_img>=128] = 255
    result_mask = result_img.clone().detach() 
    result_mask[result_mask<=128] = 0
    result_mask[result_mask>=128] = 1

    gt_canvas = torch.cat([imgs, masks], dim=-1)
    r_canvas = torch.cat([imgs, result_img], dim=-1)
    return gt_canvas, r_canvas, gt_mask, result_mask

@torch.no_grad()
def generate_sam_img(model, data, device, num, args):
    if args.data_type in ['sam_seg_coconut', 'sam_seg_cross_data', 'sam_seg_unoverlap', 'dir', 'clip', 'cosine', 'degrad', 'scale']:
        samples, refers = data['samples'], data['refer']
        sample_refers, sample_gts, refer, gt = samples['input'], samples['output'], refers['input'], refers['output']
        B, C, H, W = refer.shape
        if args.data_type == 'lvis':
            cats = data['cat']
            refer_img_ids = refers['img_id']
    elif args.data_type in ['fss', 'fss1000', 'lvis', 'seg', 'sam_seg']:
        # import pdb; pdb.set_trace()
        sample_refers, sample_gts, refer, gt = data['supp_image'], data['supp_label'], data['image'], data['label'] 
        sample_refers, sample_gts = sample_refers.transpose(0,1).contiguous(), sample_gts.transpose(0,1).contiguous()        
        B, C, H, W = refer.shape
    else:
        refer, gt, samples = data['m_img'], data['img'], data['samples']
        B, C, H, W = refer.shape

    transform = transforms.Resize(refer.shape[-2:], interpolation=transforms.InterpolationMode.NEAREST)
    refer = refer.to(device)
    gt_mask = gt.clone().detach() 

    sample_imgs = []
    if args.data_type in ['seg', 'sam_seg', 'lvis', 'sam_seg_coconut', 'sam_seg_cross_data', 'sam_seg_unoverlap', 'dir', 'clip', 'cosine', 'degrad', 'scale']:
        for i in range(len(sample_refers)):
            sample_refer = sample_refers[i].to(device)
            sample_gt = sample_gts[i].to(device)
            sample_pair = torch.cat((torch.clip(transform(sample_refer).detach().cpu() *255, 0, 255),torch.clip(transform(sample_gt).detach().cpu() *255, 0, 255)),-1)
            sample_imgs.append(sample_pair)
    elif args.data_type in ['fss', 'lvis', 'fss1000']:
        for i in range(len(sample_refers)):
            sample_refer = sample_refers[i].to(device)
            sample_gt = sample_gts[i].to(device)
            sample_pair = torch.cat((torch.clip(transform(sample_refer).detach().cpu() *255, 0, 255),torch.clip(transform(sample_gt).repeat(1, 3, 1, 1).detach().cpu() *255, 0, 255)),-1)
            sample_imgs.append(sample_pair)
    else:
        sample_refer = sample_refer.to(device)
        sample_gt = sample_gt.to(device)
        sample_pair = torch.cat((torch.clip(transform(sample_refer).detach().cpu() *255, 0, 255),torch.clip(transform(sample_gt).detach().cpu() *255, 0, 255)),-1)
        sample_imgs.append(sample_pair)
    
    if args.data_type in ['dir']: # , 'fss1000'
        sample_imgs.append(torch.cat((torch.clip(transform(refer).detach().cpu(), 0, 255), torch.clip(transform(gt).detach().cpu() *255, 0, 255)),-1)) # .repeat(1, 3, 1, 1)
    else:
        sample_imgs.append(torch.cat((torch.clip(transform(refer).detach().cpu(), 0, 255), torch.clip(transform(gt).repeat(1, 3, 1, 1).detach().cpu() *255, 0, 255)),-1)) # .repeat(1, 3, 1, 1)
    gt_canvas = torch.cat(sample_imgs, dim=-2)

    with torch.no_grad():
        if args.iou_pred:
            result_img, iou_preds = model.generate_img(data=data, data_type=args.data_type)
        else:
            result_img = model.generate_img(data=data, data_type=args.data_type, stage='train', logits_processor='softmax')
        if args.loss_analysis:
            loss_dict = model(data=data, data_type=args.data_type)
            return loss_dict
        if args.save_attn:
            attention_map = model.analysis(data=data, data_type=args.data_type, stage='train', logits_processor='softmax')
        if args.cluster_vis:
            _, cluster_masks = model.cluster_visulization(data=data, stage='train', logits_processor='softmax', return_clusters=True)
            cluster_masks = transform(cluster_masks[:, :, 0]/torch.max(cluster_masks)) # B, N, H, W
            sample_pair[..., 512:] = cluster_masks.repeat(1,3,1,1)*255
            save_path = 'analysis/cluster_mask/cvpr_coco_ade20k_518_pseudo'
            Path(save_path).mkdir(parents=True, exist_ok=True)
            save_image(sample_pair/255, f'{save_path}/{num}.png')
        if args.uncertainty_vis:
            _, support_uncertainty_maps = model.uncertainty_analysis(data=data, stage='train', logits_processor='softmax', return_uncertainty=True)
            B, C, RH, RW = gt_canvas.shape
            un_canvas = gt_canvas[..., :RH-H, :].clone().detach() 
            support_uncertainty_maps = F.interpolate(support_uncertainty_maps.squeeze().detach().cpu(), (512, 1024), mode='nearest')
            un_canvas = torch.cat([un_canvas, support_uncertainty_maps * 255], dim=-1)
            save_path = 'analysis/uncertainty_map/cvpr_coco_ade20k_518_pseudo/un_sample1' # un_sample50_all_refine_sample_epistemic
            Path(save_path).mkdir(parents=True, exist_ok=True)
            save_image(un_canvas/255, f'{save_path}/{num}.png', nrow=1)            

    # visualize pseudo mask
    if args.save_pseudo:
        refer_img = torch.clip(transform(refer).detach().cpu(), 0, 255)
        mask_img = torch.clip(transform(gt).detach().repeat(1,3,1,1).cpu(), 0, 1)
        masked_gt = refer_img*mask_img + 0.2*(refer_img*(1-mask_img))
        pm_canvas = torch.cat([(masked_gt)/255, transform(normal(result_img.detach().cpu().repeat(1,3,1,1)))], dim=-2)
        save_image(pm_canvas, f'analysis/pseudo_mask/vrp_sam_dinov2_large_random_concat_contrasive/{num}.png', normalize=True, cmap='jet', scale_each=True)

    result_img = torch.stack([torch.clip(r*255, 0, 255).repeat(3, 1, 1) for r in result_img])
    B, C, H, W = result_img.shape

    result_img[result_img>=128] = 255
    result_img[result_img<=128] = 0
    result_mask = result_img.clone().detach() 
    result_mask[result_mask<=0] = 0
    result_mask[result_mask>0] = 1

    # vis attention map
    if args.save_attn:
        cross_attn, self_attn = attention_map[0][-1], attention_map[1][-1] # [4, 16, 50, 324]

        attn_grid = []
        for attn in cross_attn:
            attn = attn.unsqueeze(1).mean(2)
            b, c, s = attn.shape
            attn_grid.append(make_grid(attn.reshape(16, 1, int(sqrt(s)), int(sqrt(s))), normalize=True, scale_each=True, nrow=4))

        attn_grid = torch.stack(attn_grid)[:,0,...] # [4, 1, 82, 82]

        img, gt_mask = torch.clip(transform(refer).detach().cpu(), 0, 255), torch.clip(transform(gt).repeat(1, 3, 1, 1).detach().cpu() *255, 0, 255)
        from matplotlib import cm

        attn_grid = np.apply_along_axis(cm.viridis, -1, attn_grid.cpu().numpy())[...,:3]
        attn_grid = F.interpolate(torch.tensor(attn_grid).permute(0,3,1,2), (512,512), mode='nearest')
        canvas = torch.cat([img/255, gt_mask/255, result_img.detach().cpu()/255, attn_grid.detach().cpu()], dim=-1)
        save_path = 'analysis/attn_map/cvpr_coco_ade20k_518_pseudo_crossattn'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        save_image(canvas, f'{save_path}/{num}.png', nrow=1)

    r_canvas = gt_canvas.clone().detach() 
    B, C, RH, RW = r_canvas.shape
    if args.data_type in ['CVF', 'test']:
        r_canvas[..., RH-H//2:,RW-H//2:] = result_img
    else:
        r_canvas[..., RH-H:,RW-H:] = result_img
    
    del refer, sample_refers, sample_gts, sample_refer, sample_gt, sample_pair
    torch.cuda.empty_cache()

    if args.iou_pred:
        return gt_canvas, r_canvas, gt_mask, result_mask, iou_preds
    else:
        return gt_canvas, r_canvas, gt_mask, result_mask

def get_test_dataset(args):

    if args.data_type == 'sam_seg':
        image_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor()])
        mask_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.Grayscale(3),
            transforms.ToTensor()])
        val_file = "/home/dmsheng/datasets/coco/annotations/val_seg_3w.json"
        # val_file = "/home/dmsheng/datasets/coco/annotations/val_seg_no_small_3w.json"
        # val_file = "/home/dmsheng/datasets/coco/annotations/val_gtav_seg_3w.json"
        # val_file = "/home/dmsheng/datasets/coco/annotations/val_sd_context_seg_3w.json"
        # val_file = "/home/dmsheng/datasets/coco/annotations/val_sd_coco_seg_3w.json"
        dataset_val = SAMSegDataset(COCO_ROOT_VAL, val_file, transform=image_transform, target_transform=mask_transform, num_samples=args.samples_num, size=args.input_size, mode='test', test_ids=args.test_ids)
        # dataset_val = SAMSegGenerationDataset(COCO_ROOT_VAL, val_file, transform=image_transform, target_transform=mask_transform, num_samples=args.samples_num, size=args.input_size)
    elif args.data_type == 'lvis':
        image_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor()])
        mask_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.Grayscale(3),
            transforms.ToTensor()])
        val_file = "/home/dmsheng/datasets/coco/annotations/lvis_img100_seg_train.json"
        dataset_val = SAMSegLVISDataset(LVIS_ROOT, LVIS_SEG_ANN_VAL, transform=image_transform, target_transform=mask_transform, num_samples=args.samples_num, size=args.input_size, fold=args.split, is_train=False)  
    elif args.data_type in ['fss', 'mix_fss']:
        image_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor()])
        mask_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.Grayscale(3),
            transforms.ToTensor()])
        # dataset_val = MyFSSCOCOFDataset(split=args.split, shot=args.samples_num, data_set='coco', base_data_root='/home/dmsheng/datasets/coco', 
        #                                 use_split_coco=True, transform=image_transform, target_transform=mask_transform, 
        #                                 mode='val')
        dataset_val = FSSCOCODataset(datapath='/home/dmsheng/datasets/coco', fold=args.split, split='test', shot=args.samples_num, transform=image_transform, target_transform=mask_transform)
    elif args.data_type in ['fss1000']:
        image_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor()])
        mask_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.Grayscale(3),
            transforms.ToTensor()])
        # dataset_val = FSS1000Dataset(shot=args.samples_num, size=args.input_size, base_data_root='/home/dmsheng/datasets/fss-1000', transform=image_transform, target_transform=mask_transform)
        dataset_val = FSS1000Dataset(base_data_root='/home/dmsheng/datasets/fss-1000', transform=image_transform, target_transform=mask_transform)
    else:
        raise TypeError

    return dataset_val

def get_args_parser():

    parser = argparse.ArgumentParser('UNICL-SAM testing', add_help=False)
    parser.add_argument('-i', '--img_path', default='figures_dataset/original_999.png', type=str)
    parser.add_argument('-m', '--model', type=str, choices=model_lists, metavar='MODEL',
                    help='Name of model to train')
    parser.add_argument('-cfg', '--config_path', 
                        default='configs/ImageGPT/ImageGPT_small_GumbelVQ_inputid.yaml', type=str)
    parser.add_argument('-pt', '--ckpt', 
                        default='experiments/ImageGPT/00001-ImageGPT_small_GumbelVQ_inputid_caption-vg-30w-epoch1000-batch64-blr0.0003-res128-samples1-weight/checkpoint-125.pth', type=str)
    parser.add_argument('-val', '--val_path', default='results/train_data', type=str)
    parser.add_argument('-l', '--val_list', default='/home/dmsheng/datasets/ILSVRC_2012/imagenetcolor_train.txt', type=str)
    parser.add_argument('-b', '--batch_size', default=25, type=int)

    parser.add_argument('-d', '--data_type', default='test', type=str, choices=data_lists)
    parser.add_argument('-s', '--input_size', default=256, type=int)
    parser.add_argument('-c', '--category', default=None)
    parser.add_argument('-n', '--samples_num', default=1, type=int)
    parser.add_argument('-o', '--obj_size', default=None)
    parser.add_argument('-t', '--select_type')
    parser.add_argument('-id', '--test_ids', default=None)
    parser.add_argument('--blank_vg_ratio', default=0.1, type=float)
    parser.add_argument('--split', default=1, type=int, choices=[0,1,2,3,4,5,6,7,8,9])

    parser.add_argument('--cluster_vis', default=False)
    parser.add_argument('--uncertainty_vis', default=False)
    parser.add_argument('--save_attn', default=False)
    parser.add_argument('--save_pseudo', default=False)
    parser.add_argument('--loss_analysis', default=False)
    parser.add_argument('--iou_pred', default=False)


    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    set_random_seed(2333)
    torch.set_grad_enabled(False)

    save_path = os.path.join('/mnt/data/homes/dmsheng/icl_seg_results/UNICL_SAM/cvpr2025', args.ckpt.split('/')[-2], args.ckpt.split('/')[-1], args.val_path)
    if args.category:
        save_path = os.path.join(save_path, args.category + '_sample' + str(args.samples_num))
    if args.obj_size:
        save_path += f'-{args.obj_size}'
    print(save_path)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.config_path)
    print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ICL_VRP_SAM_DINO_VitDet_FPN_Uncertatinty_Deterministic_Contrastive(config)
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    model.to(device)
    model.eval()

    dataset_val = get_test_dataset(args)
    print(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        shuffle=False
    )

    seg_gt_masks = []
    seg_q_gt_masks = []
    seg_re_masks = []

    id = 0
    loss = 0
    num = 0

    test_ids = []
    for i, batch in enumerate(tqdm(data_loader_val)):
        if args.data_type in ['CVF', 'test']:
            data = batch[0]
        else:
            data = batch

        if args.data_type in ['seg_class']:
            gt_canvas, r_canvas, gt_mask, result_mask = generate_seg_class_img(model, data, args, config, device, i)
            seg_gt_masks.append(gt_mask[:, 0, ...])
            seg_re_masks.append(result_mask)
            gt_canvas, r_canvas = gt_canvas.detach().cpu(), r_canvas.detach().cpu()

            # save img
            for b_id in range(gt_canvas.shape[0]):
                input_img = gt_canvas[b_id].numpy().transpose([1, 2, 0])
                input_img = input_img.astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(os.path.join(save_path, 'gt_canvas_{}.png'.format(str(i*args.batch_size+b_id).zfill(5))), input_img)

                r_img = r_canvas[b_id].numpy().transpose([1, 2, 0])
                r_img = r_img.astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(os.path.join(save_path, 'r_canvas_{}.png'.format(str(i*args.batch_size+b_id).zfill(5))), r_img)
        elif args.data_type in ['sam_seg', 'sam_seg_coconut', 'sam_seg_cross_data', 'fss', 'fss1000', 'lvis']:
            if args.iou_pred:
                gt_canvas, r_canvas, gt_mask, result_mask, cls_score = generate_sam_img(model, data, device, i, args) # , cluster_vis=True   
            else:
                gt_canvas, r_canvas, gt_mask, result_mask = generate_sam_img(model, data, device, i, args) # , cluster_vis=True   

            gt_canvas, r_canvas = gt_canvas.detach().cpu(), r_canvas.detach().cpu()

            # save img
            for b_id in range(gt_canvas.shape[0]):
                input_img = gt_canvas[b_id].numpy().transpose([1, 2, 0])
                input_img = input_img.astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(os.path.join(save_path, 'gt_canvas_{}.png'.format(str(i*args.batch_size+b_id).zfill(5))), input_img)

                r_img = r_canvas[b_id].numpy().transpose([1, 2, 0])
                r_img = r_img.astype(np.uint8)[:, :, ::-1]
                if args.iou_pred:
                    cv2.imwrite(os.path.join(save_path, 'r_canvas_{}_{}.png'.format(str(i*args.batch_size+b_id).zfill(5), round(cls_score[b_id].item(), 2))), r_img)
                else:
                    cv2.imwrite(os.path.join(save_path, 'r_canvas_{}.png'.format(str(i*args.batch_size+b_id).zfill(5))), r_img)
            
            del gt_canvas, r_canvas, gt_mask, result_mask
            torch.cuda.empty_cache()
        else:
            gt_canvas, q_canvas, r_canvas, gt_mask, q_gt_mask, result_mask = generate_img(model, model_type, data, args.data_type, device, i) # .unsqueeze(0)
            seg_gt_masks.append(gt_mask[:, 0, ...])
            seg_q_gt_masks.append(q_gt_mask[:, 0, ...])
            seg_re_masks.append(result_mask[:, 0, ...])
            gt_canvas, q_canvas, r_canvas = gt_canvas.detach().cpu(), q_canvas.detach().cpu(), r_canvas.detach().cpu()

            # save img
            for b_id in range(gt_canvas.shape[0]):
                input_img = gt_canvas[b_id].numpy().transpose([1, 2, 0])
                input_img = input_img.astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(os.path.join(save_path, 'gt_canvas_{}.png'.format(str(i*args.batch_size+b_id).zfill(5))), input_img)

                q_img = q_canvas[b_id].numpy().transpose([1, 2, 0])
                q_img = q_img.astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(os.path.join(save_path, 'q_canvas_{}.png'.format(str(i*args.batch_size+b_id).zfill(5))), q_img)

                r_img = r_canvas[b_id].numpy().transpose([1, 2, 0])
                r_img = r_img.astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(os.path.join(save_path, 'r_canvas_{}.png'.format(str(i*args.batch_size+b_id).zfill(5))), r_img)
