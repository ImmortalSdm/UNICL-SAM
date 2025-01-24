import os
from os import path, replace

import torch
import json
import numpy as np
from torchvision.datasets.vision import VisionDataset
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from tqdm import tqdm

DAVIS_ROOT = '/home/qchugroup/sdmcvpr2025/datasets/DAVIS'
YTB_ROOT = '/home/qchugroup/sdmcvpr2025/datasets/YouTube-VOS_2018_val/data/'

class YouTubeVOSTestDataset(VisionDataset):
    def __init__(self, data_root, split, 
                 transform=None,                 
                 target_transform=None,
                 transforms=None):
        super().__init__(data_root, transforms, transform, target_transform)
        # self.image_dir = path.join(data_root, 'all_frames', split+'_all_frames', 'JPEGImages')
        self.image_dir = path.join(data_root, split, 'JPEGImages')
        self.mask_dir = path.join(data_root, split, 'Annotations')

        self.vid_list = sorted(os.listdir(self.image_dir))
        self.req_frame_list = {}

        with open(path.join(data_root, split, 'meta.json')) as f:
            # read meta.json to know which frame is required for evaluation
            meta = json.load(f)['videos']

            for vid in self.vid_list:
                req_frames = []
                objects = meta[vid]['objects']
                for value in objects.values():
                    req_frames.extend(value['frames'])

                req_frames = list(set(req_frames))
                self.req_frame_list[vid] = req_frames

    def load_video(self, vid):
        image_dir_this = os.path.join(self.image_dir, vid)
        mask_dir_this = os.path.join(self.mask_dir, vid)
        frames = sorted(os.listdir(image_dir_this))
        first_gt_path = path.join(mask_dir_this, sorted(os.listdir(mask_dir_this))[0])
        
        
        frame_list = []
        for name in frames :
            frame_img = Image.open(os.path.join(image_dir_this, name)).convert('RGB')
            frame_list.append(frame_img)
        mask = Image.open(first_gt_path)
        mask = np.array(mask.convert('P'), dtype=np.uint8)
        
        return frame_list, mask, first_gt_path, frames
    
    def __getitem__(self, idx):
        # query_name, support_names, class_sample = self.sample_episode(idx)
        # query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()
        vid = self.vid_list[idx]
        frame_list, mask, mask_path, frames = self.load_video(vid)
        
        images = [self.transform(x) for x in frame_list]
        images = torch.stack(images)
        mask = torch.tensor(mask)
        mask_ids = mask.unique()[1:]
        inst_masks = mask.unsqueeze(0).expand(len(mask_ids), -1, -1) == mask_ids.view(-1, 1, 1).int()

        sample = {'images': images,
                 'inst_masks': self.target_transform(inst_masks) * 1,
                 'inst_ids': mask_ids,
                 'vid': vid,
                 'frame_ids': frames,
                 'mask_path': mask_path
                 }

        sample.update({
            'ori_inst_masks':inst_masks * 1,
        })

        return sample

    def __len__(self):
        return len(self.vid_list)

class DAVISTestDataset(VisionDataset):
    def __init__(self, data_root, 
                 imset='480p', 
                 transform=None,
                 target_transform=None,
                 transforms=None
        ):
        super().__init__(data_root, transforms, transform, target_transform)
        if False:
            self.image_dir = path.join(data_root, 'JPEGImages', 'Full-Resolution')
            self.mask_dir = path.join(data_root, 'Annotations', 'Full-Resolution')
            if not path.exists(self.image_dir):
                print(f'{self.image_dir} not found. Look at other options.')
                self.image_dir = path.join(data_root, 'JPEGImages', '1080p')
                self.mask_dir = path.join(data_root, 'Annotations', '1080p')
            assert path.exists(self.image_dir), 'path not found'
        else:
            self.image_dir = path.join(data_root, 'JPEGImages', '480p')
            self.mask_dir = path.join(data_root, 'Annotations', '480p')

        with open(path.join(data_root, 'ImageSets', imset)) as f:
            self.vid_list = sorted([line.strip() for line in f])
            # self.vid_list = sorted([line.strip().split('/')[-2] for line in f])
            # self.vid_list = sorted(list(set(self.vid_list)))

    def load_video(self, vid):
        image_dir_this = os.path.join(self.image_dir, vid)
        mask_dir_this = os.path.join(self.mask_dir, vid)
        frames = sorted(os.listdir(image_dir_this))
        first_gt_path = path.join(mask_dir_this, sorted(os.listdir(mask_dir_this))[0])
        
        frame_list = []
        for name in frames :
            frame_img = Image.open(os.path.join(image_dir_this, name)).convert('RGB')
            frame_list.append(frame_img)
        mask = Image.open(first_gt_path)
        mask = np.array(mask.convert('P'), dtype=np.uint8)
        
        return frame_list, mask, first_gt_path, frames
    
    def __getitem__(self, idx):
        vid = self.vid_list[idx]
        frame_list, mask, mask_path, frames = self.load_video(vid)
        
        images = [self.transform(x) for x in frame_list]
        images = torch.stack(images)
        mask = torch.tensor(mask)
        mask_ids = mask.unique()[1:]
        inst_masks = mask.unsqueeze(0).expand(len(mask_ids), -1, -1) == mask_ids.view(-1, 1, 1).int()

        sample = {'images': images, # N, C, H, W
                 'inst_masks': self.target_transform(inst_masks) * 1, # inst, H, W
                 'inst_ids': mask_ids, # inst
                 'vid': vid,
                 'frame_ids': frames,
                 'mask_path': mask_path
        }
        sample.update({
            'ori_inst_masks':inst_masks * 1,
        })

        return sample

    def __len__(self):
        return len(self.vid_list)
    

if __name__=='__main__':
    import random
    def set_random_seed(seed: int) -> None:
        r""" Set random seeds for reproducibility """
        if seed is None:
            seed = int(random.random() * 1e5)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    set_random_seed(6)

    image_transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor()])
    mask_transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.Grayscale(),
        transforms.ToTensor()])

    '''
    segmentaion
    '''
    seg_json_file = '/home/qchugroup/sdmcvpr2025/datasets/coco/annotations/instances_train2017.json'
    seg_image_root = '/home/qchugroup/sdmcvpr2025/datasets/coco/val2017'
    sam_seg_json = "/home/qchugroup/sdmcvpr2025/datasets/coco/annotations/train_seg_3w.json"

    seg_dataset = DAVISTestDataset(DAVIS_ROOT, imset="2017/test-dev.txt",
                                    transform=image_transform, 
                                    target_transform=mask_transform) # 'gaussian','gray','color_jitter','sobel','canny','sharp','jpeg_compression','gaussian_noise','motion_blur'

    data_loader = torch.utils.data.DataLoader(
        seg_dataset,
        batch_size=4,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )

    for num, data in tqdm(enumerate(data_loader)):
        print(data['info'])
