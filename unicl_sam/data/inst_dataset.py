import os
import pickle

from torch.utils.data import Dataset
import random
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from pycocotools.coco import COCO
from collections import defaultdict

from torchvision.transforms import Compose
from unicl_sam.data.inst_aug import ColorJitter, RandomResizedCrop, RandomApply,RandomHorizontalFlip, Norm, DeNorm

class DefaultBundle():
    def __call__(self, sample):
        if 'is_box_mask' not in sample:
            sample['is_box_mask'] = False
        if 'neg_class_names' not in sample:
            sample['neg_class_names'] = []

        return sample

class InstCOCO(Dataset):
    def __init__(self, base_image_dir, transform, is_train=True, dataset_name='coco', max_inst=10):
        self.transform = transform
        split = 'train2017' if is_train else 'val2017'
        json_path = 'annotations/instances_{}.json'.format(split)
        self.is_lvis = False
        self.is_lip = False
        self.is_box_mask = False
        if dataset_name == 'lvis' :
            self.is_lvis = True
            split = 'train2017' if is_train else 'val2017'
            split_json = 'train' if is_train else 'val'
            json_path = 'annotations/lvis_v1_{}.json'.format(split_json)
            self.img_root = base_image_dir
        elif dataset_name == 'coco' :
            split = 'train2017' if is_train else 'val2017'
            json_path = 'annotations/instances_{}.json'.format(split)
            self.img_root = os.path.join(base_image_dir, split)
        elif dataset_name == 'paco_lvis' :
            split = 'train' if is_train else 'val'
            json_path = 'annotations/paco_lvis_v1_{}.json'.format(split)
            self.img_root = os.path.join(base_image_dir)
        elif dataset_name == 'o365' :
            self.is_box_mask = True
            split = 'train' if is_train else 'val'
            json_path = 'objects365_{}.json'.format(split)
            self.img_root = os.path.join(base_image_dir, split)
        elif dataset_name == 'lip':
            self.is_lip = True
            self.img_root = os.path.join(base_image_dir, 'train_images')
            self.anno_root = os.path.join(base_image_dir, 'TrainVal_parsing_annotations/train_segmentations')
        else :
            raise NotImplementedError

        self.ids = []
        self.max_inst = max_inst
        if dataset_name == 'lip':
            for name in os.listdir(self.img_root):
                if name.endswith('.jpg'):
                    self.ids.append(name)
        else :
            self.coco = COCO(os.path.join(base_image_dir, json_path))
            ids = list(sorted(self.coco.imgs.keys()))
            for idx in ids :
                if len(self.coco.getAnnIds(idx)):
                    self.ids.append(idx)

    def __len__(self,):
        return len(self.ids)
    
    def __getitem__(self, index):
        idx = self.ids[index]
        if self.is_lip :
            image_path, idx = idx, index
            image = Image.open(os.path.join(self.img_root, image_path)).convert('RGB')
            labels = Image.open(os.path.join(self.anno_root, image_path.replace('.jpg','.png'))).convert('L')
            labels = np.array(labels)
            masks = []
            for cid in np.unique(labels)[1:]: #ignore bg
                masks.append(labels==cid)
            if len(masks) == 0:
                return self[index+1]
            masks = np.stack(masks)
        else :
            if self.is_lvis :
                coco_url = self.coco.loadImgs(idx)[0]["coco_url"]
                image_path = os.path.join(*coco_url.split('/')[-2:])
            else :
                image_path = self.coco.loadImgs(idx)[0]["file_name"]

            image = Image.open(os.path.join(self.img_root, image_path)).convert('RGB')
            annos = self.coco.loadAnns(self.coco.getAnnIds(idx))
            if self.is_box_mask :
                def bbox_to_mask(bbox):
                    x,y,w,h = bbox
                    x1,y1 = x, y
                    x2,y2 = x+w, y
                    x3,y3 = x+w, y+h
                    x4,y4 = x, y+h
                    return [[x1,y1,x2,y2,x3,y3,x4,y4]]
                for i, ann in enumerate(annos):
                    annos[i]['segmentation'] = bbox_to_mask(ann['bbox'])

            # if len(annos) > self.max_inst :
            #     annos = np.random.choice(
            #         annos, size=self.max_inst, replace=False
            #     ).tolist()
            masks = np.stack([self.coco.annToMask(x) for x in annos])

        def to_tensor(x):
            return torch.tensor(np.array(x), dtype=torch.float32).permute(2,0,1)
        image = to_tensor(image)
        masks = torch.tensor(masks).float()

        sample = {'image': image,
                 'label': masks * 255,
                #  "img_id": torch.from_numpy(np.array(idx)),
                 "shape": torch.tensor(image.shape[-2:]),
                #  "class_name": 'instance',
                 'is_inst': True,
                #  'is_box_mask': self.is_box_mask,
                #  'num_samples': torch.from_numpy(np.array(self.num_samples))
                 }

        if self.transform:
            sample = self.transform(sample)

        mask_sum = sample['supp_label'].flatten(1).sum(-1) > 100
        mask_sum_reverse = sample['label'].flatten(1).sum(-1) > 100
        mask_sum = mask_sum_reverse & mask_sum
        non_empty_idx = mask_sum.nonzero()[:,0]
        max_inst = random.randint(0, self.max_inst - 1)
        rand_idx = torch.randperm(non_empty_idx.shape[0])[:max_inst]
        select_idx = non_empty_idx[rand_idx]
        if len(select_idx) == 0 :
            select_idx = torch.tensor([0])
        sample['label'] = sample['label'][select_idx]
        sample['supp_label'] = sample['supp_label'][select_idx]

        if False :
            import cv2
            cv2.imwrite('tmp.jpg', image.permute(1,2,0).int().numpy())
            cv2.imwrite('tmp.jpg', sample['image'].permute(1,2,0).int().numpy())
            cv2.imwrite('tmp.jpg', sample['image_dual'].permute(1,2,0).int().numpy())
            pass
        return sample


def custom_collate_fn(data):
    from torch.utils.data import default_collate
    custom_collate_data = defaultdict(list)
    keys = set(data[0].keys())
    for data_this in data:
        assert set(data_this.keys()) == keys, '{} / {}'.format(set(data_this.keys())- keys, keys-set(data_this.keys()))
        for k in keys:
            if 'label' in k or k in ['neg_class_names']:
                v = data_this.get(k)
                custom_collate_data[k].append(v)
        for k in custom_collate_data.keys():
            data_this.pop(k)

    data = default_collate(data)
    data.update(**custom_collate_data)
    return data

def get_inst_aug(img_size):
    aug_list = [
                Norm(),
                RandomResizedCrop(img_size, scale=(0.3, 1.0), interpolation=3),  # 3 is bicubic
                RandomApply([
                    ColorJitter(0.4, 0.4, 0.2, 0.1)
                ], p=0.2),
                RandomHorizontalFlip(0.1),
                DeNorm(),
                # DefaultBundle()
            ]
    return Compose(aug_list)
