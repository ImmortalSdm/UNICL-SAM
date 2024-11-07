import os
import random
import numpy as np
from glob import glob
import math
import json
from tqdm import tqdm
from typing import Any
from copy import deepcopy

import albumentations as A
from albumentations import DualIAATransform, to_tuple
from albumentations.pytorch import ToTensorV2
import imgaug.augmenters as iaa

import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.transforms import functional as Func
import torch.nn.functional as F
from torchvision import transforms, utils

from torchmetrics import Metric
from scipy.signal.windows import gaussian
from PIL import ImageOps, ImageFilter

'''
Augmentations
'''
class IAAAffine2(DualIAATransform):  # 在输入上放置一个规则的点网格，并通过仿射变换随机移动这些点的邻域。
    """Place a regular grid of points on the input and randomly move the neighbourhood of these point around
    via affine transformations.

    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(
        self,
        scale=(0.7, 1.3),
        translate_percent=None,
        translate_px=None,
        rotate=0.0,
        shear=(-0.1, 0.1),
        order=1,
        cval=0,
        mode="reflect",
        always_apply=False,
        p=0.5,
    ):
        super(IAAAffine2, self).__init__(always_apply, p)
        self.scale = dict(x=scale, y=scale)
        self.translate_percent = to_tuple(translate_percent, 0)
        self.translate_px = to_tuple(translate_px, 0)
        self.rotate = to_tuple(rotate)
        self.shear = dict(x=shear, y=shear)
        self.order = order
        self.cval = cval
        self.mode = mode

    @property
    def processor(self):
        return iaa.Affine(
            self.scale,
            self.translate_percent,
            self.translate_px,
            self.rotate,
            self.shear,
            self.order,
            self.cval,
            self.mode,
        )

    def get_transform_init_args_names(self):
        return ("scale", "translate_percent", "translate_px", "rotate", "shear", "order", "cval", "mode")

class IAAPerspective2(DualIAATransform):
    """Perform a random four point perspective transform of the input.

    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}

    Args:
        scale ((float, float): standard deviation of the normal distributions. These are used to sample
            the random distances of the subimage's corners from the full image's corners. Default: (0.05, 0.1).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(self, scale=(0.05, 0.1), keep_size=True, always_apply=False, p=0.5,
                 order=1, cval=0, mode="replicate"):
        super(IAAPerspective2, self).__init__(always_apply, p)
        self.scale = to_tuple(scale, 1.0)
        self.keep_size = keep_size
        self.cval = cval
        self.mode = mode

    @property
    def processor(self):
        return iaa.PerspectiveTransform(self.scale, keep_size=self.keep_size, mode=self.mode, cval=self.cval)  # 透视变换

    def get_transform_init_args_names(self):
        return ("scale", "keep_size")

def make_odd(num):
    num = math.ceil(num)
    if num % 2 == 0:
        num += 1
    return num
    
def apply_transform(image, mask):
    strategy = [(2, 2), (0, 3), (1, 3), (3, 0), (2, 0)]
    level = 5
    transform = A.Compose([
        A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
        A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
        A.Posterize(num_bits=math.floor(8 - 0.8 * level), p=0.2 * level),
        A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
        A.GaussianBlur(blur_limit=(3, make_odd(3 + 0.8 * level)), p=0.2 * level),
        A.GaussNoise(var_limit=(2 * level, 10 * level), mean=0, per_channel=True, p=0.2 * level),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.2 * level),
        A.ImageCompression(5, 50, p=0.2 * level),
        A.Rotate(limit=4 * level, interpolation=1, border_mode=0, value=0, mask_value=None, rotate_method='largest_box',
                    crop_border=False, p=0.2 * level),
        A.HorizontalFlip(p=0.2 * level),
        A.VerticalFlip(p=0.2 * level),
        A.Affine(scale=(1 - 0.04 * level, 1 + 0.04 * level), translate_percent=None, translate_px=None, rotate=None,
                    shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                    shear={'x': (0, 2 * level), 'y': (0, 0)}
                    , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),  # x
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                    shear={'x': (0, 0), 'y': (0, 2 * level)}
                    , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)}, translate_px=None, rotate=None,
                    shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)}, translate_px=None, rotate=None,
                    shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level)
    ])
    employ = random.choice(strategy)
    level, shape = random.sample(transform[:8], employ[0]), random.sample(transform[8:], employ[1])
    img_transform = A.Compose([*level, *shape])
    random.shuffle(img_transform.transforms)
    transformed = img_transform(image=image, mask=mask)
    return transformed['image'], transformed['mask']

def get_transforms(transform_variant, out_size=None):
    if transform_variant == 'default':
        transform = A.Compose([
            A.RandomScale(scale_limit=0.2),  # +/- 20%
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),  # 限制对比度的自适应直方图均衡：增强图像的对比度的同时可以抑制噪声
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'simple':
        transform = A.Compose([
            A.RandomResizedCrop(height=out_size, width=out_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            A.RandomHorizontalFlip(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif transform_variant == 'distortions':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),  # 透视变换
            IAAAffine2(scale=(0.7, 1.3),  # 仿射变换
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size), # padding
            A.OpticalDistortion(),   #  Barrel/Pincushion 变形
            A.RandomCrop(height=out_size, width=out_size), 
            A.HorizontalFlip(),  # 翻转
            A.CLAHE(),  # 增强对比度
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),  # 随机改变亮度和对比度
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),  # 随机改变输入图像的色调、饱和度和数值
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale05_1':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.5, 1.0),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_12':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 1.2),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_07':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 0.7),  # scale 512 to 256 in average
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_light':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.02)),
            IAAAffine2(scale=(0.8, 1.8),
                       rotate=(-20, 20),
                       shear=(-0.03, 0.03)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'non_space_transform':
        transform = A.Compose([
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'seg_default':
        transform = A.Compose([
            A.Affine(scale=(0.2, 2), translate_percent=0.1, rotate=30, shear=10, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.ImageCompression(5, 50, p=0.3),
            A.GaussNoise(mean=0, var_limit=(1000, 10000), p=0.1)
        ])        
    elif transform_variant == 'color':
        transform = get_color_distortion(left=False)
    elif transform_variant == 'no_augs':
        transform = A.Compose([
            A.ToFloat()
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform

def dilate(bin_img, ksize=5):
    # 膨胀
    src_size = bin_img.numpy().shape
    pad = (ksize - 1) // 2
    # bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    out = F.interpolate(out,
                        size=src_size[2:],
                        mode="linear",
                        align_corners=True)
    return out

def erode(bin_img, ksize=5):
    # 腐蚀
    out = 1 - dilate(1 - bin_img, ksize)
    return out

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x+1e-8)
        return x

class Canny(nn.Module):
    def __init__(self, threshold=10.0):
        super(Canny, self).__init__()

        self.threshold = threshold

        filter_size = 5
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

    def forward(self, img):
        # torch.autograd.set_detect_anomaly(True)
        img_r = img[:,0:1] # [batch_size, 1, 224, 224]
        img_g = img[:,1:2]
        img_b = img[:,2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])
        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES

        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_mag = grad_mag.detach()
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159))
        grad_orientation += 180.0
        grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0
        grad_orientation = grad_orientation.detach()

        # THIN EDGES (NON-MAX SUPPRESSION)

        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        batch_size = inidices_positive.size()[0]
        pixel_count = height * width * batch_size
        pixel_range = torch.FloatTensor([range(pixel_count)])
        
        indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(batch_size, 1,height,width)

        indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(batch_size,1,height,width)

        channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        return grad_orientation.detach() # grad_mag.detach(), grad_orientation.detach(), thin_edges.detach()

class RandomResizedCrop(T.RandomResizedCrop):
    """
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """
    @staticmethod
    def get_params(img, scale, ratio):
        width, height = Func._get_image_size(img)
        area = height * width

        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        log_ratio = torch.log(torch.tensor(ratio))
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()

        return i, j, h, w

class DualAug():
    def __init__(self, aug_list, aug_list2=None):
        self.aug_list1 = transforms.Compose(aug_list)
        self.aug_list2 = transforms.Compose(aug_list2 if aug_list2 is not None else aug_list)
    
    def __call__(self, sample) -> Any:
        sample_copy = deepcopy(sample)
        sample1 = self.aug_list1(sample)
        if 'image_dual' in sample:
            imidx, image_dual, label_dual, shape = sample['imidx'], sample['image_dual'], sample['label_dual'], sample['shape']
            sample2 = {'imidx':imidx,'image':image_dual, 'label':label_dual, 'shape':shape}
            sample2 = self.aug_list2(sample2)
        else :
            sample2 = self.aug_list2(sample_copy)

        valid_keys = {'imidx','image','label','shape', 'ori_size'}
        sample1.update({'{}_dual'.format(k):v for k,v in sample2.items() if k in valid_keys})
        return sample1

class GaussianBlur(object):
    def __init__(self):
        pass

    def __call__(self, img):
        sigma = np.random.rand() * 1.9 + 0.1
        return img.filter(ImageFilter.GaussianBlur(sigma))

class Solarization(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return ImageOps.solarize(img)

def get_color_distortion(left=True):
    if left:
        p_blur = 1.0
        p_sol = 0.0
    else:
        p_blur = 0.1
        p_sol = 0.2
    # s is the strength of color distortion.
    transform = T.Compose(
        [
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                    )
                ],
                p=0.8,
            ),
            T.RandomAdjustSharpness(0.5, p=0.2),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur()], p=p_blur),
            T.RandomApply([Solarization()], p=p_sol),
        ]
    )
    return transform

'''
Segmentation
'''
def xywh_to_xyxy(boxes):
    """
    Convert [x y w h] box format to [x1 y1 x2 y2] format.
    """
    if isinstance(boxes, list):
        boxes = np.array(boxes)
    if len(boxes.shape) == 1:
        return np.hstack((boxes[0:2], boxes[0:2] + boxes[2:4] - 1))
    else:
        return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    if isinstance(boxes, list):
        boxes = np.array(boxes)
        # assert boxes.shape[1] == 4
    if len(boxes.shape) == 1:
        return np.hstack((boxes[0:2], boxes[2:4] - boxes[0:2] + 1))
    else:
        return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

def norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box

    # Calculate the normalized coordinates with min-max clamping
    norm_x1 = max(0.0, min(x1 / w, 1.0))
    norm_y1 = max(0.0, min(y1 / h, 1.0))
    norm_x2 = max(0.0, min(x2 / w, 1.0))
    norm_y2 = max(0.0, min(y2 / h, 1.0))

    # Return the normalized box coordinates
    normalized_box = [round(norm_x1, 3), round(norm_y1, 3), round(norm_x2, 3), round(norm_y2, 3)]
    return normalized_box

def de_norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    box = x1, y1, x2, y2
    return box

def expand2square(pil_img, background_color=(255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def box_xyxy_expand2square(box, *, w, h):
    if w == h:
        return box
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = x1, y1, x2, y2
        return box
    assert w < h
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = x1, y1, x2, y2
    return box

def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)
    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = abs(overlap / union)
    if exchange:
        ious = ious.T
    return ious

class MIoU(Metric):
    """Torchmetrics mean Intersection-over-Union (mIoU) implementation.

    IoU calculates the intersection area between the predicted class mask and the label class mask.
    The intersection is then divided by the area of the union of the predicted and label masks.
    This measures the quality of predicted class mask with respect to the label. The IoU for each
    class is then averaged and the final result is the mIoU score. Implementation is primarily
    based on `mmsegmentation <https://github.com/open-mmlab/mmsegmentation/blob/aa50358c71fe9c4cccdd2abe42433bdf702e757b/mmseg/core/evaluation/metrics.py#L132>`_

    Args:
        num_classes (int): the number of classes in the segmentation task.
        ignore_index (int, optional): the index to ignore when computing mIoU. Default: ``-1``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, num_classes: int, ignore_index: int = -1):
        super().__init__(dist_sync_on_step=True)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.add_state('total_intersect', default=torch.zeros(num_classes, dtype=torch.float64), dist_reduce_fx='sum')
        self.add_state('total_union', default=torch.zeros(num_classes, dtype=torch.float64), dist_reduce_fx='sum')

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update the state with new predictions and targets."""
        preds = logits.argmax(dim=1)
        for pred, target in zip(preds, targets):
            mask = (target != self.ignore_index)
            pred = pred[mask]
            target = target[mask]

            intersect = pred[pred == target]
            area_intersect = torch.histc(intersect.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
            area_prediction = torch.histc(pred.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
            area_target = torch.histc(target.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)

            self.total_intersect += area_intersect
            self.total_union += area_prediction + area_target - area_intersect

    def compute(self):
        """Aggregate state across all processes and compute final metric."""
        total_intersect = self.total_intersect[self.total_union != 0]  # type: ignore (third-party)
        total_union = self.total_union[self.total_union != 0]  # type: ignore (third-party)
        return 100 * (total_intersect / total_union).mean()

'''
Others
'''
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy()
        elif isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

def value2key(data_dict, data):
    return list(data_dict.keys())[list(data_dict.values()).index(data)]

def get_context_anno():
    imgs = sorted(list(glob('/home/dmsheng/datasets/sd_context/image/*', recursive=True)))
    masks = sorted(list(glob('/home/dmsheng/datasets/sd_context/heatmap/*.png', recursive=True)))

    coco_json = "/home/dmsheng/datasets/coco/annotations/val_seg_3w.json"
    import json
    with open(coco_json,'r') as fr: 
        coco_json = json.load(fr) 

    for k,v in coco_json['per_cat_pool'].items():
        if coco_id_name_map[int(k)] not in context_coco_overlap_map.values():
            for id in coco_json['per_cat_pool'][str(k)]:
                coco_json['dataset_dicts'].pop(str(id))

    num = 0
    dataset_dicts = {}
    for k,v in coco_json['dataset_dicts'].items():
        dataset_dicts[num] = v
        num += 1

    per_cat_pool={int(k): [] for k,v in coco_json['per_cat_pool'].items()}
    for i, img in enumerate(imgs):
        context_name = img.split('/')[-1].split('_')[0]
        if context_name in context_coco_overlap_map.keys():
            coco_name = context_coco_overlap_map[context_name]
            coco_id = [k for k,v in coco_id_name_map.items() if v == coco_name][0]
            per_cat_pool[coco_id].append({
                'img': img,
                'mask': masks[i]
            })

    annos = {}
    annos['dataset_dicts'] = dataset_dicts
    annos['per_cat_pool'] = per_cat_pool

    with open('/home/dmsheng/datasets/coco/annotations/val_sd_context_seg_3w.json', 'w', encoding='utf-8') as f:
        json.dump(annos, f, ensure_ascii=False)

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0,
                          filter_value: float = -float('Inf')) -> torch.Tensor:
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
        From: https://arxiv.org/abs/1904.09751
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        :param logits:
        :param filter_value:
        :param top_p:
        :param top_k:
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits
