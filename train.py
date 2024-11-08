# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import time
from pathlib import Path
import cv2
cv2.setNumThreads(0)
import torch
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision import datasets
from torch.distributed.elastic.multiprocessing.errors import record
import timm.optim.optim_factory as optim_factory

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.misc import get_model
from utils.args import model_lists, data_lists
from unicl_sam.data import *
from unicl_sam.utils.engine import *

from omegaconf import OmegaConf

def get_args_parser():
    parser = argparse.ArgumentParser('ImageGPT pre-training', add_help=False)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int, # 64
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vrp_sam_dino', type=str, choices=model_lists, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--config', default='configs/unicl_sam/vrp_sam_dinov2_large.yaml', type=str,
                        help='Config of model')
    parser.add_argument('--input_size', default=518, type=int,
                        help='images input size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--break_after_epoch', type=int, metavar='N', help='break training after X epochs, to tune hyperparams and avoid messing with training schedule')

    # Dataset parameters
    parser.add_argument('--data_type', default='CVF', type=str, choices=data_lists)
    parser.add_argument('--data_path', default='/shared/yossi_gandelsman/arxiv/arxiv_data/', type=str,
                        help='dataset path')
    parser.add_argument('--data_name', default=None, type=str)
    parser.add_argument('--samples_num', default=1, type=int)
    parser.add_argument('--split', default=1, type=int, choices=[0,1,2,3,4,5,6,7,8,9])
    parser.add_argument('--subsample', action='store_true')
    parser.set_defaults(subsample=False)
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def get_train_dataset(args):

    if args.data_type == 'sam_seg_imgiter':
        image_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor()])
        mask_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.Grayscale(3),
            transforms.ToTensor()])
        dataset_train = SAMImgSegDataset(transform=image_transform, target_transform=mask_transform, num_samples=args.samples_num, size=args.input_size)
        dataset_val = MixSemSegCOCODataset(transform=image_transform, target_transform=mask_transform, num_samples=args.samples_num, size=args.input_size, stage='val')
    elif args.data_type == 'sam_seg_imgiter_degrad_contrastive':
        image_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor()])
        mask_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.Grayscale(3),
            transforms.ToTensor()])
        dataset_train = SAMImgSegContrastiveDataset(root=COCO_ROOT_TRAIN, transform=image_transform, target_transform=mask_transform, num_samples=args.samples_num, size=args.input_size)
        dataset_val = MixSemSegCOCODataset(transform=image_transform, target_transform=mask_transform, num_samples=args.samples_num, size=args.input_size, stage='val')
    elif args.data_type == 'sam_seg_semantic_degrad_contrastive':
        image_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor()])
        mask_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.Grayscale(3),
            transforms.ToTensor()])
        dataset_train = MixSemSegContrastiveDataset(transform=image_transform, target_transform=mask_transform, num_samples=args.samples_num, size=args.input_size)
        dataset_val = MixSemSegCOCODataset(transform=image_transform, target_transform=mask_transform, num_samples=args.samples_num, size=args.input_size, stage='val')
    elif args.data_type in ['fss', 'mix_fss']:
        image_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor()])
        mask_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.Grayscale(3),
            transforms.ToTensor()])
        dataset_train = FSS1000Dataset(split=args.split, shot=args.samples_num, data_set='coco', base_data_root='/home/qchugroup/sdmcvpr2025/datasets/coco', 
                                        use_split_coco=True, transform=image_transform, target_transform=mask_transform, 
                                        mode='train')
        dataset_val = FSS1000Dataset(split=args.split, shot=args.samples_num, data_set='coco', base_data_root='/home/qchugroup/sdmcvpr2025/datasets/coco', 
                                        use_split_coco=True, transform=image_transform, target_transform=mask_transform, 
                                        mode='val')
    else:
        raise TypeError

    return dataset_train, dataset_val

@record
def main(args):
    misc.init_distributed_mode(args)

    desc = args.config.split('/')[-1].split('.')[0]

    desc += f'-{args.model}'
    desc += f'-{args.data_type}'
    if args.data_type == 'calibrate_vg':
        desc += f'-prob{args.blank_vg_ratio:g}'
    desc += f'-epoch{args.epochs:d}'
    desc += f'-batch{args.batch_size:d}'
    desc += f'-blr{args.blr:g}'
    desc += f'-res{args.input_size:d}'
    desc += f'-samples{args.samples_num:d}'

    print('Experiments name:', desc)

    prev_run_dirs = []
    if os.path.isdir(args.output_dir):
        prev_run_dirs = [x for x in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(args.output_dir, f'{cur_run_id:05d}-{desc}')

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    misc.set_random_seed(seed)

    cudnn.benchmark = True

    dataset_train, dataset_val = get_train_dataset(args)
    print(dataset_train)
    print('Length of dataset: ', len(dataset_train))
    print('Length of val dataset: ', len(dataset_val))

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if args.data_type == 'mix_seg_vg':
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    log_dir = os.path.join(args.run_dir, 'log_dir')
    log_writer = SummaryWriter(log_dir)


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size, # for sam
        num_workers=args.num_workers, 
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True
    )

    '''
    define the model
    '''
    config = OmegaConf.load(args.config)
    config['args'] = vars(args)
    print(config)
    OmegaConf.save(config=config, f=os.path.join(args.run_dir, 'config.yaml'))

    model = get_model(args, config)
    model.to(device)
    
    # model.tokenizer.img_tokenizer.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    total_params += sum(p.numel() for p in model.buffers())
    print(f'{total_params/(1024*1024):.2f}M total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params/(1024*1024):.2f}M training parameters.')
    print("Model = %s" % str(model))

    epoch_size = len(dataset_train)
    print(f'epoch_size is {epoch_size}')
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    print('eff_batch_size:', eff_batch_size)
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    base_lr = (args.lr * 256 / eff_batch_size)
    print("base lr: %.2e" % base_lr)
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    for k, v in model_without_ddp.named_parameters():
        if 'vqgan' in k:
            v.requires_grad = False
        if 'tokenizer' in k:
            v.requires_grad = False

    for n,p in model.named_parameters():
        if p.requires_grad: print(n)

    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    if args.data_type == 'mix_seg_vg':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    print(optimizer)

    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            epoch_size=epoch_size // eff_batch_size
        )
        if args.output_dir and (epoch % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs): # args.save_ckpt_freq
            validate(model, data_loader_val, device, epoch, log_writer=log_writer, args=args)
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {'epoch': epoch,
                    **{f'train_{k}': v for k, v in train_stats.items()}}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.run_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    # cv2.setNumThreads(0)
    mp.set_start_method('spawn')
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
