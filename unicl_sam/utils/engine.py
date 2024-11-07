# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import os
import sys
from typing import Iterable
import numpy as np
import torch

from utils.metrics import compute_iou
from utils.logger import get_logger
import utils.misc as misc
import utils.lr_sched as lr_sched
from utils.meter import AverageMeter

datasets_name_map = {
    'COCO_INC_S': 'MySegDataset',
    'VG_INC_CAP': 'MyVGDataset',
    'vg_bbox_text': 'MyVGBboxTextDataset'
}

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    epoch_size=1):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    data_loader_i = iter(data_loader)
    # import pdb; pdb.set_trace()

    for data_iter_step in metric_logger.log_every(range(epoch_size*accum_iter), print_freq, header):
        if args.data_type == 'CVF':
            (batch, _) = next(data_loader_i)
        else:
            batch = next(data_loader_i)

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            loss_dict = model(batch, args.data_type)

        loss = torch.stack([v for k, v in loss_dict.items() if 'gate' not in k]).sum() # scale
        loss_value = loss.detach().item()

        metric_logger.update(**{k: v.detach().item() for k, v in loss_dict.items()})

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        loss /= accum_iter # caption and vg

        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 20)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def validate(model, data_loader, device, epoch, log_writer, args):
    # logger = get_logger(name='mmdet', log_file=os.path.join(args.run_dir, "meter.txt"), file_mode='a')
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 40
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    eval_meter = AverageMeter([0,1], logger=None)

    if args.data_type == 'CVF':
        for data_iter_step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            with torch.cuda.amp.autocast():
                loss_dict, _ = model(batch, args.data_type)

            loss_plus_value = loss_dict['CE_plus']

            if not math.isfinite(loss_plus_value):
                print("Loss is {}, stopping training".format(loss_plus_value))
                sys.exit(1)

            metric_logger.update(**{'val_' + k: v.item() for k, v in loss_dict.items()})
            del loss_dict
    elif args.data_type in ['vp', 'mix_seg_vp', 'mix_seg_corrupt', 'mix_seg_vg', 'calibrate_vg']:
        for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            with torch.cuda.amp.autocast():
                # loss_dict, loss_data_dict, _ = model(batch, args.data_type)
                # loss_dict, _ = model(batch, args.data_type)
                loss_dict, loss_data_dict, _, _ = model(batch, args.data_type)

            loss_plus_value = loss_dict['CE_plus']

            if not math.isfinite(loss_plus_value):
                print("Loss is {}, stopping training".format(loss_plus_value))
                sys.exit(1)

            metric_logger.update(**{'val_' + k: v for k, v in loss_dict.items()})
            metric_logger.update(**{'val_' + k: v for k, v in loss_data_dict.items()})
            del loss_dict
            # metric_logger.update(**{'val_' + k: v for k, v in loss_data_dict.items()})   
    else:
        for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            with torch.cuda.amp.autocast():
                output, loss_dict = model(batch, args.data_type, infer=True, return_logits=False)

            loss = torch.stack([v for k, v in loss_dict.items() if 'graph' not in k]).sum()
            loss_value = loss.detach().item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            inter, union = compute_iou(output['pred'], output['gt'], True)
            eval_meter.update(inter.cuda(), union.cuda(), 1)

            metric_logger.update(**{'val_' + k: v.detach().item() for k, v in loss_dict.items()})

            del loss_dict, loss

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats for val:", metric_logger)
    eval_meter.write_result(header=header)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

