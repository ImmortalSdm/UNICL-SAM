from collections import defaultdict
import copy
import math
import sys
import os

sys.path.append('.')

from glob import glob
import logging
import random
import cv2
import torch
import json
import bisect
import numpy as np
from PIL import Image
from copy import deepcopy
from torch.utils.data import Dataset, ConcatDataset
from tempfile import NamedTemporaryFile
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch.nn.functional import interpolate
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from tqdm import tqdm
from panopticapi.utils import rgb2id

from unicl_sam.data.data_utils import *
from unicl_sam.data.data_setting import *
from unicl_sam.model import ResizeLongestSide

logger = logging.getLogger(__name__) # don't ask me why this import works


'''
Segmentation
'''
def get_segs(img_info, segm):
    if isinstance(segm, dict):
        if isinstance(segm["counts"], list):
            # convert to compressed RLE
            segm = mask_util.frPyObjects(segm, *segm["size"])
    else:   
        # filter out invalid polygons (< 3 points)
        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
        if len(segm) == 0:
            raise ValueError(
                f"No valid segmentaion for {img_info['id']}."
            )
        segm = mask_util.frPyObjects(segm, img_info["height"], img_info["width"])
        segm = mask_util.merge(segm)
    
    return segm

def get_img_class_mask(coco_api, img_id):
    img_info = coco_api.loadImgs(img_id)[0]
    cat_ids = set([x["category_id"] for index, x in enumerate(coco_api.imgToAnns[img_id]) if {k: v for k, v in x.items() if (k == "category_id")}])
    anns = coco_api.imgToAnns[img_id]

    img_path = img_info['file_name']
    img = cv2.imread(os.path.join(f'/home/qchugroup/sdmcvpr2025/datasets/coco/train2017/{img_path}'))

    save_path = f'cvpr_pics/test/{img_id}'
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(f'{save_path}/test_{img_id}.png', img)
    for id in cat_ids:
        segs = [x["segmentation"] for index, x in enumerate(anns) if {k: v for k, v in x.items() if (k == "category_id" and v == id)}]

        segs = [get_segs(img_info, poly) for poly in segs]
        # segs = [mask_util.merge(poly) for poly in segs]
        segs = mask_util.merge(segs)
        masks = mask_util.decode([segs]).repeat(3, axis=-1)* 255
        cv2.imwrite(f'{save_path}/test_{img_id}_{id}.png', masks)

def get_class_mask(coco_api, cat_id):
    cat_ids = coco_api.getImgIds(catIds=[cat_id])
    cat_name = coco_api.cats[cat_id]
    save_path = f'cvpr_pics/{cat_name}'
    os.makedirs(save_path, exist_ok=True)

    for i in tqdm(range(20)):
        img_id = random.choice(cat_ids)
        img_info = coco_api.loadImgs(img_id)[0]
        img_path = img_info['file_name']
        img = cv2.imread(os.path.join(f'/home/qchugroup/sdmcvpr2025/datasets/coco/train2017/{img_path}'))
        cv2.imwrite(f'{save_path}/test_{img_id}.png', img)

        anns = coco_api.imgToAnns[img_id]
        segs = [x["segmentation"] for index, x in enumerate(anns) if {k: v for k, v in x.items() if (k == "category_id" and v == cat_id)}]
        segs = [get_segs(img_info, poly) for poly in segs]
        segs = mask_util.merge(segs)
        masks = mask_util.decode([segs]).repeat(3, axis=-1)* 255
        cv2.imwrite(f'{save_path}/test_{img_id}_mask.png', masks)

def load_coco_seg_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None, img_size=128, size_type=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO
    from fvcore.common.timer import Timer
    from detectron2.utils.file_io import PathManager
    import contextlib
    import io
    
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = {}
    img_cat_dicts = {}
    person_img_ids = coco_api.getImgIds(catIds=[1])
    laptop_img_ids = coco_api.getImgIds(catIds=[73])

    per_cat_pool={int(i): [] for i, cat in coco_id_name_map.items()}

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id", "area"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    record_id = 0 # id for each anno
    for (img_dict, anno_dict_list) in imgs_anns:
        img_id = img_dict['id']
        # if img_id in person_img_ids:
        #     continue
        # else:
        cats = len(set([x["category_id"] for index, x in enumerate(coco_api.imgToAnns[img_id]) if {k: v for k, v in x.items() if (k == "category_id")}]))
        img_cat_dicts[img_id] = cats

        if min(img_dict["height"], img_dict["width"]) < img_size:
            continue

        for cat_id, cat in coco_id_name_map.items(): 
            cat_anns = [x for index, x in enumerate(anno_dict_list) if 
            {k: v for k, v in x.items() if (k == "category_id" and v == cat_id)}
            ]
            if len(cat_anns) > 0:
                record = {}
                record["file_name"] = os.path.join(image_root, img_dict["file_name"])
                record["height"] = img_dict["height"]
                record["width"] = img_dict["width"]
                image_id = record["image_id"] = img_dict["id"]
                objs = []
                for anno in cat_anns:
                    # get different coco size object
                    if size_type == 'small':
                        if anno["area"] >= 32*32:
                            continue
                    elif size_type == 'medium':
                        if anno["area"] < 32*32 or anno["area"] > 96*96:
                            continue
                    elif size_type == 'large':
                        if anno["area"] <= 96*96:
                            continue

                    # Check that the image_id in this annotation is the same as
                    # the image_id we're looking at.
                    # This fails only when the data parsing logic or the annotation file is buggy.

                    # The original COCO valminusminival2014 & minival2014 annotation files
                    # actually contains bugs that, together with certain ways of using COCO API,
                    # can trigger this assertion.
                    assert anno["image_id"] == image_id

                    assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

                    obj = {key: anno[key] for key in ann_keys if key in anno}
                    if "bbox" in obj and len(obj["bbox"]) == 0:
                        raise ValueError(
                            f"One annotation of image {image_id} contains empty 'bbox' value! "
                            "This json does not have valid COCO format."
                        )

                    segm = anno.get("segmentation", None)
                    if segm:  # either list[list[float]] or dict(RLE)
                        if isinstance(segm, dict):
                            if isinstance(segm["counts"], list):
                                # convert to compressed RLE
                                segm = mask_util.frPyObjects(segm, *segm["size"])
                        else:   
                            # filter out invalid polygons (< 3 points)
                            segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                            if len(segm) == 0:
                                num_instances_without_valid_segmentation += 1
                                continue  # ignore this instance
                            segm = mask_util.frPyObjects(segm, img_dict["height"], img_dict["width"])
                            segm = mask_util.merge(segm)
                        obj["segmentation"] = segm

                    keypts = anno.get("keypoints", None)
                    if keypts:  # list[int]
                        for idx, v in enumerate(keypts):
                            if idx % 3 != 2:
                                # COCO's segmentation coordinates are floating points in [0, H or W],
                                # but keypoint coordinates are integers in [0, H-1 or W-1]
                                # Therefore we assume the coordinates are "pixel indices" and
                                # add 0.5 to convert to floating point coordinates.
                                keypts[idx] = v + 0.5
                        obj["keypoints"] = keypts

                    if id_map:
                        annotation_category_id = obj["category_id"]
                        try:
                            obj["category_id"] = id_map[annotation_category_id]
                        except KeyError as e:
                            raise KeyError(
                                f"Encountered category_id={annotation_category_id} "
                                "but this id does not exist in 'categories' of the json file."
                            ) from e
                    objs.append(obj)
                if len(objs) == 0:
                    continue
                record["annotations"] = objs
                record["category_id"] = cat_id
                per_cat_pool[cat_id].append(record_id)
                dataset_dicts[record_id] = record
                record_id += 1

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )

    # data = {}
    # data['dataset_dicts'] = dataset_dicts
    # data['per_cat_pool'] = per_cat_pool
    # # data['per_img_unoverlap_pool'] = part_per_img_unoverlap_pool
    # with open(os.path.join('/home/qchugroup/sdmcvpr2025/datasets/coco/annotations/', "train_seg_35w.json"), 'w', encoding='utf-8') as f:
    #     json.dump(data, f, cls=MyEncoder, ensure_ascii=False)  # , cls=MyEncoder

    img_cat_dicts = sorted(img_cat_dicts.items(),key=lambda x:x[1], reverse=True)
    img_cat_dicts = [info[0] for info in img_cat_dicts]

    for i in tqdm(range(100)):
        img_id = img_cat_dicts[i]
        get_img_class_mask(coco_api, img_id)

    import pdb; pdb.set_trace()

    return dataset_dicts, per_cat_pool

def load_coco_img_seg_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None, img_size=128, size_type=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO
    from fvcore.common.timer import Timer
    from detectron2.utils.file_io import PathManager
    import contextlib
    import io
    
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'flickr_url': 'http://farm4.staticflickr.com/3612/3348961791_a229740c7c_z.jpg', 
    # 'id': 581921, 
    # 'neg_category_ids': [475, 604, 669, 1006, 1192, 673], 
    # 'not_exhaustive_category_ids': [], 
    # 'width': 640, 
    # 'license': 1, 
    # 'coco_url': 'http://images.cocodataset.org/train2017/000000581921.jpg', 
    # 'date_captured': '2013-11-20 13:14:15', 
    # 'height': 427}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'bbox': [336.92, 535.54, 5.8, 12.52], 
    # 'category_id': 1023, 
    # 'image_id': 185826, 
    # 'id': 1270141, 
    # 'segmentation': [[338.04, 537.97, 341.6, 548.06, 342.72, 546.38, 340.85, 542.46, 337.67, 535.54, 336.92, 535.73, 338.04, 537.97]], 
    # 'area': 13.48},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = {}
    img_cat_dicts = {}

    per_cat_pool={int(i): [] for i, cat in coco_id_name_map.items()}

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id", "area"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    record_id = 0 # id for each anno
    for (img_dict, anno_dict_list) in imgs_anns:
        img_id = img_dict['id']

        cats = [x["category_id"] for index, x in enumerate(coco_api.imgToAnns[img_id]) if {k: v for k, v in x.items() if (k == "category_id")}]
        if len(cats) == 0:
            num_instances_without_valid_segmentation += 1
            continue
        img_cat_dicts[img_id] = {
            'cats': cats,
            'size': [img_dict["width"], img_dict["height"]],
            "file_name": os.path.join(image_root, img_dict["file_name"]),
            "image_id": img_id
        }

        for cat_id, cat in coco_id_name_map.items(): 
            cat_anns = [x for index, x in enumerate(anno_dict_list) if 
            {k: v for k, v in x.items() if (k == "category_id" and v == cat_id)}
            ]
            if len(cat_anns) > 0:
                objs = []
                for anno in cat_anns:
                    # get different coco size object
                    if size_type == 'small':
                        if anno["area"] >= 32*32:
                            continue
                    elif size_type == 'medium':
                        if anno["area"] < 32*32 or anno["area"] > 96*96:
                            continue
                    elif size_type == 'large':
                        if anno["area"] <= 96*96:
                            continue

                    assert anno["image_id"] == img_id

                    assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

                    obj = {key: anno[key] for key in ann_keys if key in anno}
                    if "bbox" in obj and len(obj["bbox"]) == 0:
                        raise ValueError(
                            f"One annotation of image {img_id} contains empty 'bbox' value! "
                            "This json does not have valid COCO format."
                        )

                    objs.append(obj)
                if len(objs) == 0:
                    continue
                per_cat_pool[cat_id].append(img_id)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )

    return coco_api, img_cat_dicts, per_cat_pool

class SAMSegDataset(VisionDataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 512

    def __init__(
        self,
        root,
        annFile,
        transform=None,
        target_transform=None,
        transforms=None,
        size=128, 
        num_samples=1,
        cat=1,
        obj_size=None, 
        mode='train',
        test_ids=None
    ):
        super().__init__(root, transforms, transform, target_transform)

        # self.coco, self.ds, self.per_cat_pool, self.id_ann_map = load_coco_json(annFile, root)
        with open(annFile,'r') as fr: 
            seg_json = json.load(fr) 

        self.ds, self.per_cat_pool = np.array(seg_json['dataset_dicts']), np.array(seg_json['per_cat_pool'])
        if 'cat_pool_dicts' in seg_json.keys():
            self.cat_pool = np.array(seg_json['cat_pool_dicts'])
        else:
            self.cat_pool = None
        self.mode = mode

        if self.mode == 'test':
            with open(test_ids,'r') as fr: 
                test_ids = json.load(fr) 
            self.test_ids = test_ids

        self.img_size = size
        self.num_samples = num_samples
        self.cat = cat
        self.scale = (0.1, 1)
        self.ratio = (3.0/4.0, 4.0/3.0)
        self.crop = RandomResizedCrop(self.img_size, scale=self.scale, ratio=self.ratio)
        self.sam_transform = ResizeLongestSide(self.img_size)
        # self.sam_transform = T.ToTensor()
        self.output_transform = T.Resize((self.img_size, self.img_size), antialias=True)

        del seg_json
        # print(self.ds)

    def get_crop(self, img, shape, bbox, seg, cat, input=False): # , scale=(1.2, 3), ratio=(3.0/4.0, 4.0/3.0)
        img_w, img_h = shape
        x,y,w,h = random.choice(bbox)
        x,y,w,h = np.floor(x), np.floor(y), np.ceil(w), np.ceil(h)

        img_symbol = 0
        if img_w >= img_h:
            img_min_len = img_h
        else:
            img_min_len = img_w
            img_symbol = 1

        if w >= h:
            bbox_max_len = w
        else:
            bbox_max_len = h

        seg = mask_util.merge(seg)
        mask = mask_util.decode([seg])
        # mask += (mask == 1) * (m * (cat - 1))
    
        if img_min_len >= bbox_max_len:
            crop_len = crop_w = crop_h = img_min_len
            crop_x = random.randint(int(max(0, x+w-crop_len)), int(min(x, img_w-crop_len)))
            crop_y = random.randint(int(max(0, y+h-crop_len)), int(min(y, img_h-crop_len)))
        else:
            crop_x = random.randint(0, int(x))
            crop_y = random.randint(0, int(y))
            crop_x2 = random.randint(int(x + w), img_w)
            crop_y2 = random.randint(int(y + h), img_h)
            crop_w = crop_x2 - crop_x
            crop_h = crop_y2 - crop_y

        crop_img = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :]
        crop_mask = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :]
        if input:
            # return Image.fromarray(crop_img).resize((self.img_size, self.img_size)), Image.fromarray(crop_mask[..., 0]).resize((self.img_size, self.img_size))
            return cv2.resize(crop_img, (self.img_size, self.img_size)), cv2.resize(crop_mask[..., 0], (self.img_size, self.img_size))
        else:
            return Image.fromarray(crop_img), Image.fromarray(crop_mask.repeat(3, axis=-1)*255)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def get_refer_record(self, cat_id, img_id):
        try:
            ids = random.sample(self.per_cat_pool[()][str(cat_id)], k=self.num_samples+1)
        except:
            ids = random.choices(self.per_cat_pool[()][str(cat_id)], k=self.num_samples+1)

        if self.cat_pool:
            sample_records = [self.cat_pool[()][str(id)] for id in ids]
        else:
            sample_records = [self.ds[()][str(id)] for id in ids]
        img_ids = [re['image_id'] for re in sample_records]

        if img_id in img_ids:
            idx = img_ids.index(img_id)
            ids.pop(idx)
            sample_records.pop(idx)

        return sample_records[:self.num_samples]

    def mask_transform(self, mask):
        if isinstance(mask, np.ndarray):
            if len(mask.shape) == 3:
                mask = Image.fromarray(np.squeeze(mask, axis=-1))
            else:
                mask = Image.fromarray(mask)
        return torch.LongTensor(np.array(self.output_transform(mask)).astype('int8'))

    def __getitem__(self, index):
        if self.mode == 'test':
            refer_record = self.ds[()][str(self.test_ids[index]['target'])]
        else:
            refer_record = self.ds[()][str(index)]

        refer_cat_id = refer_record['category_id']
        refer_img_id = refer_record['image_id']
        refer_img_path = refer_record["file_name"]
        refer_img_shape = [refer_record["width"], refer_record["height"]]
        refer_segs = [ann["segmentation"] for ann in refer_record["annotations"]]
        refer_bboxs = [ann["bbox"] for ann in refer_record["annotations"]] # x,y,w,h

        refer_img = Image.open(refer_img_path).convert("RGB")
        refer_crop_img, refer_crop_mask = self.get_crop(np.array(refer_img), refer_img_shape, refer_bboxs, refer_segs, refer_cat_id, input=True)
        if self.transforms is not None:
            refer_crop_img = self.sam_transform.apply_image(refer_crop_img)
            refer_crop_img = torch.tensor(refer_crop_img).permute(2, 0, 1)
            refer_crop_mask = self.mask_transform(refer_crop_mask).unsqueeze(0)

        samples_input = []
        samples_output = []
        samples_img_id = []

        if self.mode == 'test':
            if self.cat_pool:
                sample_records = [self.cat_pool[()][str(id)] for id in self.test_ids[index]['sample']]
            else:
                sample_records = [self.ds[()][str(id)] for id in self.test_ids[index]['sample']]
        else:
            sample_records = self.get_refer_record(refer_cat_id, refer_img_id)
        
        for i in range(self.num_samples):
            sample_record = sample_records[i]
            sample_img_id = sample_record['image_id']
            sample_img_path = sample_record["file_name"]
            sample_img_shape = [sample_record["width"], sample_record["height"]]
            sample_segs = [ann["segmentation"] for ann in sample_record["annotations"]]
            sample_bboxs = [ann["bbox"] for ann in sample_record["annotations"]]

            sample_img = Image.open(sample_img_path).convert("RGB")
            sample_crop_img, sample_crop_mask = self.get_crop(np.array(sample_img), sample_img_shape, sample_bboxs, sample_segs, refer_cat_id)

            if self.transforms is not None:
                sample_crop_img, sample_crop_mask = self.transforms(sample_crop_img, sample_crop_mask)
            
            samples_input.append(sample_crop_img)
            samples_output.append(sample_crop_mask)
            samples_img_id.append(sample_img_id)

        sample = {
            'image': refer_crop_img,
            'label': refer_crop_mask,
            'supp_image': torch.stack(samples_input),
            'supp_label': torch.stack(samples_output),
            # "img_id": torch.from_numpy(np.array(refer_img_id)),
            # 'supp_img_id': torch.tensor(samples_img_id),
            # "shape": torch.tensor(refer_img_shape),
            # "class_name": torch.from_numpy(np.array(int(refer_cat_id))),
            'is_inst': False,
            'info': 'COCO_INC_S',
            'num_samples': torch.from_numpy(np.array(self.num_samples))
        }

        return sample

    def __len__(self):
        return len(self.ds[()])

class SAMImgSegDataset(VisionDataset):
    '''
    iter per image
    '''
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    def __init__(
        self,
        transform=None,
        target_transform=None,
        transforms=None,
        size=128, 
        num_samples=1,
        cat=1,
        is_train=True, 
        dataset_name='coco_img_iter', 
    ):
        if is_train:
            root = COCO_ROOT_TRAIN
            annFile = COCO_SEG_ANN_TRAIN
        else:
            root = COCO_ROOT_VAL
            annFile = COCO_SEG_ANN_VAL

        super().__init__(root, transforms, transform, target_transform)

        self.coco_api, self.img_cat_dicts, self.per_cat_pool = load_coco_img_seg_json(annFile, root)
        self.keys = list(self.img_cat_dicts.keys())

        self.dataset_name = dataset_name
        self.img_size = size
        self.num_samples = num_samples
        self.cat = cat
        self.scale = (0.1, 1)
        self.ratio = (3.0/4.0, 4.0/3.0)
        self.crop = RandomResizedCrop(self.img_size, scale=self.scale, ratio=self.ratio)
        self.sam_transform = ResizeLongestSide(self.img_size)
        self.output_transform = T.Resize((self.img_size, self.img_size), antialias=True)

    def get_seg(self, segm, size):
        height, width = size[1], size[0]
        if isinstance(segm, dict):
            if isinstance(segm["counts"], list):
                # convert to compressed RLE
                segm = mask_util.frPyObjects(segm, *segm["size"])
        else:   
            # filter out invalid polygons (< 3 points)
            segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
            segm = mask_util.frPyObjects(segm, height, width)
            segm = mask_util.merge(segm)
        
        return segm

    def get_crop(self, img, bboxs, segs):
        while True:
            crop_x, crop_y, crop_w, crop_h = self.crop.get_params(img, scale=self.scale, ratio=self.ratio)

            crop_bbox = np.array([crop_x, crop_y, crop_w, crop_h])

            overlaps = bbox_overlaps(xywh_to_xyxy(crop_bbox)[None, :], xywh_to_xyxy(np.array(bboxs)), mode = 'iou')
            # print(overlaps)
            if np.sum(overlaps) > 0:
                break
        
        segs = mask_util.merge(segs)
        mask = mask_util.decode(segs).repeat(3, axis=-1)* 255

        crop_img = np.array(img)[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :]
        crop_mask = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :]
        return Image.fromarray(crop_img), Image.fromarray(crop_mask)

    def old_get_crop(self, img, shape, bbox, seg, cat, input=False): # , scale=(1.2, 3), ratio=(3.0/4.0, 4.0/3.0)
        img_w, img_h = shape
        x,y,w,h = random.choice(bbox)
        x,y,w,h = np.floor(x), np.floor(y), np.ceil(w), np.ceil(h)

        img_symbol = 0
        if img_w >= img_h:
            img_min_len = img_h
        else:
            img_min_len = img_w
            img_symbol = 1

        if w >= h:
            bbox_max_len = w
        else:
            bbox_max_len = h
        
        seg = mask_util.merge(seg)
        mask = mask_util.decode([seg])
        # mask += (mask == 1) * (m * (cat - 1))
    
        if img_min_len >= bbox_max_len:
            crop_len = crop_w = crop_h = img_min_len
            crop_x = random.randint(int(max(0, x+w-crop_len)), int(min(x, img_w-crop_len)))
            crop_y = random.randint(int(max(0, y+h-crop_len)), int(min(y, img_h-crop_len)))
        else:
            crop_x = random.randint(0, int(x))
            crop_y = random.randint(0, int(y))
            crop_x2 = random.randint(int(x + w), img_w)
            crop_y2 = random.randint(int(y + h), img_h)
            crop_w = crop_x2 - crop_x
            crop_h = crop_y2 - crop_y

        crop_img = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :]
        crop_mask = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :]
        if input:
            # return Image.fromarray(crop_img).resize((self.img_size, self.img_size)), Image.fromarray(crop_mask[..., 0]).resize((self.img_size, self.img_size))
            return cv2.resize(crop_img, (self.img_size, self.img_size)), cv2.resize(crop_mask[..., 0], (self.img_size, self.img_size))
        else:
            return Image.fromarray(crop_img), Image.fromarray(crop_mask.repeat(3, axis=-1)*255)

    def _mask_transform(self, mask):
        if isinstance(mask, np.ndarray):
            if len(mask.shape) == 3:
                mask = Image.fromarray(np.squeeze(mask, axis=-1))
            else:
                mask = Image.fromarray(mask)
        return torch.LongTensor(np.array(self.output_transform(mask)).astype('int32'))

    def __getitem__(self, index):
        refer_record = self.img_cat_dicts[self.keys[index]]
        refer_cat_id = random.choice(refer_record['cats'])
        refer_img_id = refer_record['image_id']
        refer_img_path = refer_record["file_name"]
        refer_img_shape = refer_record['size']
        refer_anns = self.coco_api.loadAnns(self.coco_api.getAnnIds(imgIds=[refer_img_id], catIds=[refer_cat_id]))
        refer_segs = [self.get_seg(ann["segmentation"], refer_img_shape) for ann in refer_anns]
        refer_bboxs = [ann["bbox"] for ann in refer_anns] # x,y,w,h
        # import pdb; pdb.set_trace()

        refer_img = Image.open(refer_img_path).convert("RGB")
        refer_crop_img, refer_crop_mask = self.old_get_crop(np.array(refer_img), refer_img_shape, refer_bboxs, refer_segs, refer_cat_id, input=True)
        if self.transforms is not None:
            refer_crop_img = self.sam_transform.apply_image(refer_crop_img)
            refer_crop_img = torch.tensor(refer_crop_img).float().permute(2, 0, 1)
            refer_crop_mask = self._mask_transform(refer_crop_mask).float().unsqueeze(0)

        samples_input = []
        samples_output = []
        samples_img_id = []
        for i in range(self.num_samples):
            sample_record = self.img_cat_dicts[random.choice(self.per_cat_pool[refer_cat_id])]
            sample_img_id = sample_record['image_id']
            sample_img_path = sample_record["file_name"]
            sample_img_shape = sample_record['size']
            sample_anns = self.coco_api.loadAnns(self.coco_api.getAnnIds(imgIds=[sample_img_id], catIds=[refer_cat_id]))
            sample_segs = [self.get_seg(ann["segmentation"], sample_img_shape) for ann in sample_anns]
            sample_bboxs = [ann["bbox"] for ann in sample_anns]

            sample_img = Image.open(sample_img_path).convert("RGB")
            sample_crop_img, sample_crop_mask = self.old_get_crop(np.array(sample_img), sample_img_shape, sample_bboxs, sample_segs, refer_cat_id)

            if self.transforms is not None:
                # sample_crop_img = self.sam_transform.apply_image(np.array(sample_crop_img))
                # sample_crop_img = torch.from_numpy(sample_crop_img).permute(2, 0, 1).contiguous()
                # sample_crop_mask = self.target_transform(sample_crop_mask)
                sample_crop_img, sample_crop_mask = self.transforms(sample_crop_img, sample_crop_mask)
            
            samples_input.append(sample_crop_img)
            samples_output.append(sample_crop_mask)
            samples_img_id.append(sample_img_id)

        sample = {
            'image': refer_crop_img,
            'label': refer_crop_mask*255,
            'supp_image': torch.stack(samples_input)[0]*255,
            'supp_label': torch.stack(samples_output)[0]*255,
            # "img_id": torch.from_numpy(np.array(refer_img_id)),
            # 'supp_img_id': torch.tensor(samples_img_id),
            "shape": torch.tensor(refer_img_shape),
            # "class_name": torch.from_numpy(np.array(int(refer_cat_id))),
            'is_inst': False,
            # 'info': self.dataset_name,
            # 'num_samples': torch.from_numpy(np.array(self.num_samples))
        }

        return sample

    def __len__(self):
        return len(self.img_cat_dicts)

class SAMImgSegContrastiveDataset(VisionDataset):
    '''
    iter per image
    '''
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        transforms=None,
        is_train=True, 
        dataset_name='coco_img_iter', 
        size=128, 
        num_samples=1,
        cat=1,
        select_type=None,
        label_select_type=None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        if is_train:
            annFile = COCO_SEG_ANN_TRAIN
        else:
            annFile = COCO_SEG_ANN_VAL

        self.coco_api, self.img_cat_dicts, self.per_cat_pool = load_coco_img_seg_json(annFile, root)
        self.keys = list(self.img_cat_dicts.keys())

        self.dataset_name = dataset_name
        self.img_size = size
        self.num_samples = num_samples
        self.cat = cat
        self.select_type = select_type
        self.label_select_type = label_select_type

        self.sam_transform = ResizeLongestSide(self.img_size)
        self.output_transform = T.Resize((self.img_size, self.img_size), antialias=True)

    def get_seg(self, segm, size):
        height, width = size[1], size[0]
        if isinstance(segm, dict):
            if isinstance(segm["counts"], list):
                # convert to compressed RLE
                segm = mask_util.frPyObjects(segm, *segm["size"])
        else:   
            # filter out invalid polygons (< 3 points)
            segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
            segm = mask_util.frPyObjects(segm, height, width)
            segm = mask_util.merge(segm)
        
        return segm

    def get_crop(self, img, bboxs, segs):
        while True:
            crop_x, crop_y, crop_w, crop_h = self.crop.get_params(img, scale=self.scale, ratio=self.ratio)

            crop_bbox = np.array([crop_x, crop_y, crop_w, crop_h])

            overlaps = bbox_overlaps(xywh_to_xyxy(crop_bbox)[None, :], xywh_to_xyxy(np.array(bboxs)), mode = 'iou')
            # print(overlaps)
            if np.sum(overlaps) > 0:
                break
        
        segs = mask_util.merge(segs)
        mask = mask_util.decode(segs).repeat(3, axis=-1)* 255

        crop_img = np.array(img)[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :]
        crop_mask = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :]
        return Image.fromarray(crop_img), Image.fromarray(crop_mask)

    def old_get_crop(self, img, shape, bbox, seg, cat, input=False): # , scale=(1.2, 3), ratio=(3.0/4.0, 4.0/3.0)
        img_w, img_h = shape
        x,y,w,h = random.choice(bbox)
        x,y,w,h = np.floor(x), np.floor(y), np.ceil(w), np.ceil(h)

        img_symbol = 0
        if img_w >= img_h:
            img_min_len = img_h
        else:
            img_min_len = img_w
            img_symbol = 1

        if w >= h:
            bbox_max_len = w
        else:
            bbox_max_len = h
        
        seg = mask_util.merge(seg)
        mask = mask_util.decode([seg])
        # mask += (mask == 1) * (m * (cat - 1))
    
        if img_min_len >= bbox_max_len:
            crop_len = crop_w = crop_h = img_min_len
            crop_x = random.randint(int(max(0, x+w-crop_len)), int(min(x, img_w-crop_len)))
            crop_y = random.randint(int(max(0, y+h-crop_len)), int(min(y, img_h-crop_len)))
        else:
            crop_x = random.randint(0, int(x))
            crop_y = random.randint(0, int(y))
            crop_x2 = random.randint(int(x + w), img_w)
            crop_y2 = random.randint(int(y + h), img_h)
            crop_w = crop_x2 - crop_x
            crop_h = crop_y2 - crop_y

        crop_img = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :]
        crop_mask = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :]
        if input:
            # return Image.fromarray(crop_img).resize((self.img_size, self.img_size)), Image.fromarray(crop_mask[..., 0]).resize((self.img_size, self.img_size))
            return cv2.resize(crop_img, (self.img_size, self.img_size)), cv2.resize(crop_mask[..., 0], (self.img_size, self.img_size))
        else:
            return Image.fromarray(crop_img), Image.fromarray(crop_mask.repeat(3, axis=-1)*255)

    def _mask_transform(self, mask):
        if isinstance(mask, np.ndarray):
            if len(mask.shape) == 3:
                mask = Image.fromarray(np.squeeze(mask, axis=-1))
            else:
                mask = Image.fromarray(mask)
        return torch.LongTensor(np.array(self.output_transform(mask)).astype('int32'))

    def get_refer_record(self, cat_id, img_id):
        try:
            ids = random.sample(self.per_cat_pool[cat_id], k=self.num_samples+1)
        except:
            ids = random.choices(self.per_cat_pool[cat_id], k=self.num_samples+1)

        sample_records = [self.img_cat_dicts[id] for id in ids]
        img_ids = [re['image_id'] for re in sample_records]

        if img_id in img_ids:
            idx = img_ids.index(img_id)
            sample_records.pop(idx)
            img_ids.pop(idx)

        return sample_records[:self.num_samples], img_ids[:self.num_samples]

    def __getitem__(self, index):
        refer_record = self.img_cat_dicts[self.keys[index]]
        refer_cat_id = random.choice(refer_record['cats'])
        refer_img_id = refer_record['image_id']
        refer_img_path = refer_record["file_name"]
        refer_img_shape = refer_record['size']
        refer_anns = self.coco_api.loadAnns(self.coco_api.getAnnIds(imgIds=[refer_img_id], catIds=[refer_cat_id]))
        refer_segs = [self.get_seg(ann["segmentation"], refer_img_shape) for ann in refer_anns]
        refer_bboxs = [ann["bbox"] for ann in refer_anns] # x,y,w,h
        # import pdb; pdb.set_trace()

        refer_img = Image.open(refer_img_path).convert("RGB")
        refer_crop_img, refer_crop_mask = self.old_get_crop(np.array(refer_img), refer_img_shape, refer_bboxs, refer_segs, refer_cat_id, input=True)
        if self.transforms is not None:
            refer_crop_img = self.sam_transform.apply_image(refer_crop_img)
            refer_crop_img = torch.tensor(refer_crop_img).permute(2, 0, 1)
            refer_crop_mask = self._mask_transform(refer_crop_mask).unsqueeze(0)

        if self.label_select_type == None:
            # select_type = random.choice(list(self.degradation_pools.keys()))
            label_select_type = random.choice(["bbox", "dilate", "erode", "None"])
        else:
            # select_type = self.select_type
            label_select_type = self.label_select_type

        samples_input = []
        samples_output = []
        samples_ctr_input = []
        samples_ctr_output = []

        sample_records, _ = self.get_refer_record(refer_cat_id, refer_img_id)
        for i in range(self.num_samples):
            sample_record = sample_records[i]
            sample_img_id = sample_record['image_id']
            sample_img_path = sample_record["file_name"]
            sample_img_shape = sample_record['size']
            sample_anns = self.coco_api.loadAnns(self.coco_api.getAnnIds(imgIds=[sample_img_id], catIds=[refer_cat_id]))
            sample_segs = [self.get_seg(ann["segmentation"], sample_img_shape) for ann in sample_anns]
            sample_bboxs = [ann["bbox"] for ann in sample_anns]

            sample_img = Image.open(sample_img_path).convert("RGB")
            sample_crop_img, sample_crop_mask = self.old_get_crop(np.array(sample_img), sample_img_shape, sample_bboxs, sample_segs, refer_cat_id)
            sample_ctr_image, sample_ctr_mask = deepcopy(sample_crop_img), deepcopy(sample_crop_mask) 

            for i in range(3):
                try:
                    temp_img, temp_mask = apply_transform(np.array(sample_ctr_image), np.array(sample_ctr_mask))
                    temp_mask = (temp_mask > 0).astype(temp_img.dtype)
                    assert temp_mask.max() > 0
                    sample_ctr_image, sample_ctr_mask = Image.fromarray(temp_img), Image.fromarray(temp_mask)
                    break
                except:
                    pass

            if self.transforms is not None:
                sample_crop_img, sample_crop_mask = self.transforms(sample_crop_img, sample_crop_mask)
                sample_ctr_image, sample_ctr_mask = self.transforms(sample_ctr_image, sample_ctr_mask)

            if label_select_type == 'dilate':
                sample_ctr_mask = dilate(sample_ctr_mask, ksize=10)
                sample_ctr_mask = self.output_transform(sample_ctr_mask)
            elif label_select_type == 'erode':
                sample_ctr_mask = erode(sample_ctr_mask, ksize=10)                
                sample_ctr_mask = self.output_transform(sample_ctr_mask)
            elif label_select_type == 'bbox':
                try:
                    idx = torch.where(sample_ctr_mask > 0)
                    y_min, y_max = torch.min(idx[1], dim=0).item(), torch.max(idx[1], dim=0).item()
                    x_min, x_max = torch.min(idx[2], dim=0).item(), torch.max(idx[2], dim=0).item()

                    sample_ctr_mask = torch.zeros_like(sample_ctr_mask)
                    sample_ctr_mask[:, y_min:y_max, x_min:x_max] = 1  
                except:
                    pass
            else:
                pass          
            
            samples_input.append(sample_crop_img)
            samples_ctr_input.append(sample_ctr_image)
            samples_output.append(sample_crop_mask)
            samples_ctr_output.append(sample_ctr_mask)

        sample = {
            'image': refer_crop_img,
            'label': refer_crop_mask,
            'supp_image': torch.stack(samples_input),
            'supp_label': torch.stack(samples_output),
            'supp_ctr_image': torch.stack(samples_ctr_input),
            'supp_ctr_label': torch.stack(samples_ctr_output),
            # "img_id": torch.from_numpy(np.array(refer_img_id)),
            # 'supp_img_id': torch.tensor(samples_img_id),
            # "shape": torch.tensor(refer_img_shape),
            # "class_name": torch.from_numpy(np.array(int(refer_cat_id))),
            'is_inst': False,
            'info': self.dataset_name,
            'num_samples': torch.from_numpy(np.array(self.num_samples))
        }

        return sample

    def __len__(self):
        return len(self.img_cat_dicts)

class SAMSegLVISDataset(VisionDataset):
    def __init__(self, 
                 root,
                 annFile=None,
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 is_train=True, 
                 dataset_name='lvis', 
                 custom_json_path=None,
                 size=128, 
                 num_samples=1,
                 fold=None,
                 is_semseg=False, 
                 ext='png', 
                 is_meta=False,
                 is_contra=False):
        super().__init__(root, transforms, transform, target_transform)

        self.dataset_name = dataset_name
        self.is_lvis = True
        self.transform = transform
        self.is_semseg = is_semseg
        self.is_meta = is_meta
        self.is_train = is_train
        self.is_contra = is_contra
        self.num_samples = num_samples
        self.img_size = size
        self.fold = fold
        self.sam_transform = ResizeLongestSide(self.img_size)
        self.output_transform = T.Resize((self.img_size, self.img_size), antialias=True)
        if annFile is not None:
            self.img_root = root
        else:
            split_json = 'train' if is_train else 'val'
            annFile = 'annotations/lvis_v1_{}.json'.format(split_json)
            self.img_root = root

            if custom_json_path is not None :
                annFile = custom_json_path

        if fold is not None:
            assert is_train == False
            assert fold < 10

            with open(os.path.join(root, annFile)) as f:
                lvis_json = json.load(f)
                cat_count = defaultdict(set)
                for info in lvis_json['annotations']:
                    cat_count[info['category_id']].add(info['image_id'])
            cats = [x for x,v in cat_count.items() if len(v) > 1]
            cats = sorted(cats)

            idx = range(fold, len(cats), 10)
            cats = [cats[x] for x in idx]
            cats_set = set(cats)
            new_annotations = [x for x in lvis_json['annotations'] if x['category_id'] in cats_set]
            new_image_ids = set([x['image_id'] for x in new_annotations])
            new_images = [x for x in lvis_json['images'] if x['id'] in new_image_ids]
            lvis_json['annotations'] = new_annotations
            lvis_json['images'] = new_images
            with NamedTemporaryFile('w+t') as f:
                json.dump(lvis_json, f)
                print('filename is:', f.name)
                f.flush()
                self.coco = COCO(
                    f.name
                )
        else :
            self.coco = COCO(os.path.join(root, annFile))

        ids = list(sorted(self.coco.imgs.keys()))
        self.ids = []
        self.max_inst = 1

        self.cid_to_cat = {k:v['name'] for k,v in self.coco.cats.items()}
        self.cls_to_idx = defaultdict(list)
        self.class_ids = self.get_class_ids()
        for idx in tqdm(ids) :
            if len(self.coco.getAnnIds(idx)):
                self.ids.append(idx)
                annos = self.coco.loadAnns(self.coco.getAnnIds(idx))
                cat_ids = np.array([x['category_id'] for x in annos])
                uni_cats = np.unique(cat_ids)
                for cid in uni_cats:
                    self.cls_to_idx[cid].append(len(self.ids)-1) # append this idx

    def _get_ref_cid(self, cid, index):
        idx_list = self.cls_to_idx[cid]
        ref_index = index
        idx_list = list(set(idx_list) - {index})
        if len(idx_list) > 1 :
            ref_index = random.choice(idx_list)
        return ref_index

    def __len__(self,):
        if self.fold is not None and not self.is_train:
            if self.is_lvis:
                return 2300
            return max(len(self.ids) * 5, 100)
        return len(self.ids)
    
    def get_class_ids(self,):
        return np.array(sorted(self.coco.cats.keys()))

    def get_class_names(self,):
        cls_ids = sorted(self.coco.cats.keys())
        return [self.coco.cats[x]['name'] for x in cls_ids]

    def mask_transform(self, mask):
        if isinstance(mask, np.ndarray):
            if len(mask.shape) == 3:
                mask = Image.fromarray(np.squeeze(mask, axis=-1))
            else:
                mask = Image.fromarray(mask)
        return torch.LongTensor(np.array(self.output_transform(mask)).astype('int8'))

    def __getitem__(self, index):
        if self.fold is not None and not self.is_train:
            index = random.randint(0, len(self.ids) - 1)
            # self.ids[index]
        
        def _get_info(index, cats_list=None):
            idx = self.ids[index]
            if self.is_lvis:
                coco_url = self.coco.loadImgs(idx)[0]["coco_url"]
                image_path = os.path.join(*coco_url.split('/')[-2:])
            else :
                image_path = self.coco.loadImgs(idx)[0]["file_name"]

            image = Image.open(os.path.join(self.img_root, image_path)).convert('RGB')
            annos = self.coco.loadAnns(self.coco.getAnnIds(idx))
            masks = np.stack([self.coco.annToMask(x) for x in annos])
            cat_ids = np.array([x['category_id'] for x in annos])
            uni_cats = np.unique(cat_ids)
            masks_list = []
            if cats_list is None :
                if len(uni_cats) > self.max_inst :
                    cats_list = np.random.choice(
                        uni_cats, size=self.max_inst, replace=False
                    ).tolist()
                else :
                    cats_list = uni_cats.tolist()

            for cat in cats_list :
                masks_list.append(masks[cat_ids==cat].max(0))

            masks = np.stack(masks_list)

            return image, np.squeeze(masks), cats_list

        if self.fold is not None and not self.is_train:
            cats_list = [random.choice(list(self.cls_to_idx.keys()))]
            index = self._get_ref_cid(cats_list[0], None)
            image, masks, _ = _get_info(index, cats_list)
        else:
            image, masks, cats_list = _get_info(index)

        if self.transforms is not None:
            image = self.sam_transform.apply_image(np.array(image))
            image = torch.tensor(image).permute(2, 0, 1)
            image = self.output_transform(image).float()
            masks = self.mask_transform(masks).float().unsqueeze(0)

        support_imgs, support_masks = [], []
        for cid in cats_list:
            ref_index = self._get_ref_cid(cid, index)
            image_ref, masks_ref, _ = _get_info(ref_index, [cid])

            if self.transforms is not None:
                image_ref, masks_ref = self.transforms(image_ref, Image.fromarray(masks_ref*255))
            support_imgs.append(image_ref)
            support_masks.append(masks_ref)

        masks_ori = masks
        sample = {
            'image': image,
            'label': masks*255,
            'supp_image': torch.stack(support_imgs)[0]*255,
            'supp_label': torch.stack(support_masks)[0]*255,
            'is_inst': False,
            # 'info': self.dataset_name,
            # "img_id": torch.from_numpy(np.array(index)),
            "shape": torch.tensor(image.shape[-2:]),
            # 'num_samples': torch.from_numpy(np.array(self.num_samples)),
        }

        if self.is_meta:
            sample.update(class_id=cats_list[0])

        if self.is_semseg:
            all_cats = np.array(sorted(self.coco.cats.keys()))
            img_id = self.ids[index]
            annos = self.coco.loadAnns(self.coco.getAnnIds(img_id))
            masks = np.stack([self.coco.annToMask(x) for x in annos]) #(ninst, h, w)
            cat_ids = np.array([x['category_id'] for x in annos]) #(ninst, )
            all_cats_this = cat_ids[None] == all_cats[:, None]
 
            all_cats_this = cat_ids[None] == all_cats[:, None]
            ninst, h, w = masks.shape
            semmask = (all_cats_this @ masks.reshape(ninst, -1)).reshape(-1, h, w).clip(max=1)
            sample['origin_semmask'] = semmask

        # if not self.is_train:
        #     sample.update({
        #         'ori_label':masks_ori,
        #         'class_id': torch.tensor(cats_list[0])
        #     })


        if False :
            import cv2
            cv2.imwrite('tmp.jpg', image.permute(1,2,0).int().numpy())
            cv2.imwrite('tmp.jpg', sample['image'].permute(1,2,0).int().numpy())
            cv2.imwrite('tmp.jpg', sample['image_dual'].permute(1,2,0).int().numpy())
            pass
        return sample

class SAMSegLVISContrastiveDataset(VisionDataset):
    def __init__(self, 
                 root,
                 annFile=None,
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 is_train=True, 
                 dataset_name='lvis', 
                 custom_json_path=None,
                 size=128, 
                 num_samples=1,
                 fold=None,
                 is_semseg=False, 
                 ext='png', 
                 label_select_type=None,
                 is_meta=False,
                 is_contra=False):
        super().__init__(root, transforms, transform, target_transform)

        self.dataset_name = dataset_name
        self.is_lvis = True
        self.transform = transform
        self.is_semseg = is_semseg
        self.is_meta = is_meta
        self.is_train = is_train
        self.is_contra = is_contra
        self.num_samples = num_samples
        self.img_size = size
        self.fold = fold
        self.label_select_type = label_select_type
        self.sam_transform = ResizeLongestSide(self.img_size)
        self.output_transform = T.Resize((self.img_size, self.img_size), antialias=True)
        if annFile is not None:
            self.img_root = root
        else:
            split_json = 'train' if is_train else 'val'
            annFile = 'annotations/lvis_v1_{}.json'.format(split_json)
            self.img_root = root

            if custom_json_path is not None :
                annFile = custom_json_path

        if fold is not None:
            assert is_train == False
            assert fold < 10

            with open(os.path.join(root, annFile)) as f:
                lvis_json = json.load(f)
                cat_count = defaultdict(set)
                for info in lvis_json['annotations']:
                    cat_count[info['category_id']].add(info['image_id'])
            cats = [x for x,v in cat_count.items() if len(v) > 1]
            cats = sorted(cats)

            idx = range(fold, len(cats), 10)
            cats = [cats[x] for x in idx]
            cats_set = set(cats)
            new_annotations = [x for x in lvis_json['annotations'] if x['category_id'] in cats_set]
            new_image_ids = set([x['image_id'] for x in new_annotations])
            new_images = [x for x in lvis_json['images'] if x['id'] in new_image_ids]
            lvis_json['annotations'] = new_annotations
            lvis_json['images'] = new_images
            with NamedTemporaryFile('w+t') as f:
                json.dump(lvis_json, f)
                print('filename is:', f.name)
                f.flush()
                self.coco = COCO(
                    f.name
                )
        else :
            self.coco = COCO(os.path.join(root, annFile))

        ids = list(sorted(self.coco.imgs.keys()))
        self.ids = []
        self.max_inst = 1

        self.cid_to_cat = {k:v['name'] for k,v in self.coco.cats.items()}
        self.cls_to_idx = defaultdict(list)
        self.class_ids = self.get_class_ids()
        for idx in tqdm(ids) :
            if len(self.coco.getAnnIds(idx)):
                self.ids.append(idx)
                annos = self.coco.loadAnns(self.coco.getAnnIds(idx))
                cat_ids = np.array([x['category_id'] for x in annos])
                uni_cats = np.unique(cat_ids)
                for cid in uni_cats:
                    self.cls_to_idx[cid].append(len(self.ids)-1) # append this idx

    def _get_ref_cid(self, cid, index):
        idx_list = self.cls_to_idx[cid]
        ref_index = index
        idx_list = list(set(idx_list) - {index})
        if len(idx_list) > 1 :
            ref_index = random.choice(idx_list)
        return ref_index

    def __len__(self,):
        if self.fold is not None and not self.is_train:
            if self.is_lvis:
                return 2300
            return max(len(self.ids) * 5, 100)
        return len(self.ids)
    
    def get_class_ids(self,):
        return np.array(sorted(self.coco.cats.keys()))

    def get_class_names(self,):
        cls_ids = sorted(self.coco.cats.keys())
        return [self.coco.cats[x]['name'] for x in cls_ids]

    def mask_transform(self, mask):
        if isinstance(mask, np.ndarray):
            if len(mask.shape) == 3:
                mask = Image.fromarray(np.squeeze(mask, axis=-1))
            else:
                mask = Image.fromarray(mask)
        return torch.LongTensor(np.array(self.output_transform(mask)).astype('int8'))

    def __getitem__(self, index):
        if self.fold is not None and not self.is_train:
            index = random.randint(0, len(self.ids) - 1)
            # self.ids[index]
        
        def _get_info(index, cats_list=None, supp=False):
            idx = self.ids[index]
            if self.is_lvis:
                coco_url = self.coco.loadImgs(idx)[0]["coco_url"]
                image_path = os.path.join(*coco_url.split('/')[-2:])
            else :
                image_path = self.coco.loadImgs(idx)[0]["file_name"]

            image = Image.open(os.path.join(self.img_root, image_path)).convert('RGB')
            annos = self.coco.loadAnns(self.coco.getAnnIds(idx))
            masks = np.stack([self.coco.annToMask(x) for x in annos])
            cat_ids = np.array([x['category_id'] for x in annos])
            uni_cats = np.unique(cat_ids)
            masks_list = []
            if cats_list is None :
                if len(uni_cats) > self.max_inst :
                    cats_list = np.random.choice(
                        uni_cats, size=self.max_inst, replace=False
                    ).tolist()
                else :
                    cats_list = uni_cats.tolist()

            for cat in cats_list :
                masks_list.append(masks[cat_ids==cat].max(0))

            masks = np.stack(masks_list)

            if supp:
                image = np.array(image)
                masks = masks.astype(image.dtype)

            return image, np.squeeze(masks), cats_list

        if self.fold is not None and not self.is_train:
            cats_list = [random.choice(list(self.cls_to_idx.keys()))]
            index = self._get_ref_cid(cats_list[0], None)
            image, masks, _ = _get_info(index, cats_list)
        else:
            image, masks, cats_list = _get_info(index)

        if self.transforms is not None:
            image = self.sam_transform.apply_image(np.array(image))
            image = torch.tensor(image).permute(2, 0, 1)
            image = self.output_transform(image)
            masks = self.mask_transform(masks).unsqueeze(0)

        if self.label_select_type == None:
            label_select_type = random.choice(["bbox", "dilate", "erode", "None"])
        else:
            label_select_type = self.label_select_type

        support_imgs, support_masks = [], []
        support_ctr_imgs, support_ctr_masks = [], []
        for cid in cats_list:
            ref_index = self._get_ref_cid(cid, index)
            image_ref, masks_ref, _ = _get_info(ref_index, [cid], supp=True)
            image_ref_ctr = deepcopy(image_ref) 
            masks_ref_ct = deepcopy(masks_ref)

            for i in range(3):
                try:
                    temp_img, temp_mask = apply_transform(image_ref_ctr, masks_ref_ct)
                    temp_mask = (temp_mask > 0).astype(temp_img.dtype)
                    assert temp_mask.max() > 0
                    image_ref_ctr, masks_ref_ct = temp_img, temp_mask
                    break
                except:
                    pass

            if self.transforms is not None:
                image_ref, masks_ref = self.transforms(Image.fromarray(image_ref), Image.fromarray(masks_ref))
                image_ref_ctr, masks_ref_ct = self.transforms(Image.fromarray(image_ref_ctr), Image.fromarray(masks_ref_ct))

            if label_select_type == 'dilate':
                masks_ref_ct = dilate(masks_ref_ct, ksize=10)
                masks_ref_ct = self.output_transform(masks_ref_ct)
            elif label_select_type == 'erode':
                masks_ref_ct = erode(masks_ref_ct, ksize=10)                
                masks_ref_ct = self.output_transform(masks_ref_ct)
            elif label_select_type == 'bbox':
                try:
                    idx = torch.where(masks_ref_ct > 0)
                    y_min, y_max = torch.min(idx[1], dim=0)[0].item(), torch.max(idx[1], dim=0)[0].item()
                    x_min, x_max = torch.min(idx[2], dim=0)[0].item(), torch.max(idx[2], dim=0)[0].item()

                    masks_ref_ct = torch.zeros_like(masks_ref_ct)
                    masks_ref_ct[:, y_min:y_max, x_min:x_max] = 1  
                except:
                    pass
            else:
                pass   

            support_imgs.append(image_ref)
            support_masks.append(masks_ref)
            support_ctr_imgs.append(image_ref_ctr)
            support_ctr_masks.append(masks_ref_ct)

        masks_ori = masks

        sample = {
            'image': image,
            'label': masks,
            'supp_image': torch.stack(support_imgs),
            'supp_label': torch.stack(support_masks),
            'supp_ctr_image': torch.stack(support_ctr_imgs),
            'supp_ctr_label': torch.stack(support_ctr_masks),
            'is_inst': False,
            'info': self.dataset_name,
            # "imidx": torch.from_numpy(np.array(index)),
            # "shape": torch.tensor(image.shape[-2:]),
            'num_samples': torch.from_numpy(np.array(self.num_samples)),
        }

        if self.is_meta:
            sample.update(class_id=cats_list[0])

        if self.is_semseg:
            all_cats = np.array(sorted(self.coco.cats.keys()))
            img_id = self.ids[index]
            annos = self.coco.loadAnns(self.coco.getAnnIds(img_id))
            masks = np.stack([self.coco.annToMask(x) for x in annos]) #(ninst, h, w)
            cat_ids = np.array([x['category_id'] for x in annos]) #(ninst, )
            all_cats_this = cat_ids[None] == all_cats[:, None]
 
            all_cats_this = cat_ids[None] == all_cats[:, None]
            ninst, h, w = masks.shape
            semmask = (all_cats_this @ masks.reshape(ninst, -1)).reshape(-1, h, w).clip(max=1)
            sample['origin_semmask'] = semmask

        if not self.is_train:
            sample.update({
                'ori_label':masks_ori,
                'class_id': torch.tensor(cats_list[0])
            })


        if False :
            import cv2
            cv2.imwrite('tmp.jpg', image.permute(1,2,0).int().numpy())
            cv2.imwrite('tmp.jpg', sample['image'].permute(1,2,0).int().numpy())
            cv2.imwrite('tmp.jpg', sample['image_dual'].permute(1,2,0).int().numpy())
            pass
        return sample

class SAMSegADEDataset(VisionDataset):
    def __init__(self, 
                 root,
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 is_train=True, 
                 dataset_name='ade20k', 
                 size=128, 
                 num_samples=1,
                 is_semseg=False, 
                 ext='png', 
                 is_meta=False):
        super().__init__(root, transforms, transform, target_transform)

        self.img_size = size
        self.num_samples = num_samples
        self.ext = ext
        self.split = split = 'training' if is_train else 'validation'
        self.is_semseg = is_semseg
        self.zero_start = False
        self.is_meta = is_meta
        self.dataset_name = dataset_name
        self.sam_transform = ResizeLongestSide(self.img_size)
        self.output_transform = T.Resize((self.img_size, self.img_size), antialias=True)
        classes_name_list = ADE_SEM_CLASSES
        if dataset_name in ('ade20k'):
            self.ignore_idx = 255
            self.img_root = os.path.join(root, "images", split)
            self.anno_root = os.path.join(root, "annotations", split)
            classes_name_list = ['none'] + classes_name_list
        elif dataset_name in ('sd_ade20k'):
            self.ignore_idx = 255
            self.img_root = os.path.join(root, "images_detectron2", split)
            self.anno_root = os.path.join(root, "annotations_detectron2", split)
            classes_name_list = ['none'] + classes_name_list
        elif dataset_name == 'cocostuff':
            self.img_root = os.path.join(root, "images", split)
            self.anno_root = os.path.join(root, "annotations", split)
            classes_name_list = [x.split(':')[-1] for x in classes_name_list]
        elif dataset_name == 'ade847':
            self.ignore_idx = 65535
            self.img_root = os.path.join(root, "images_detectron2", split)
            self.anno_root = os.path.join(root, "annotations_detectron2", split)
            classes_name_list = ['none'] + classes_name_list
        elif dataset_name == 'sd_ade847':
            self.zero_start = False
            self.ignore_idx = 65535
            self.img_root = os.path.join(root, "images_detectron2", split)
            self.anno_root = os.path.join(root, "annotations_detectron2", split)
            classes_name_list = ['none'] + classes_name_list
        elif dataset_name == 'pc459':
            assert not is_train
            self.zero_start = True
            self.split = split = 'training' if is_train else 'validation'
            self.img_root = os.path.join(root, 'JPEGImages')
            self.anno_root = os.path.join(root, 'annotations_detectron2/pc459_val')
        else :
            raise NotImplementedError

        self.cid_to_cat = np.array(classes_name_list)
        self.all_cls = set(list(range(len(self.cid_to_cat)))) 
        if not self.zero_start:
            self.all_cls = self.all_cls- {0}
        self.class_ids = self.get_class_ids()
        file_names = sorted(
            os.listdir(self.img_root)
        )
        
        gt_names = os.listdir(self.anno_root)
        if len(file_names) != len(gt_names):
            print('warning, not equal')
            file_names = [x[:-len('.jpg')] for x in file_names]
            gt_names = [x[:-(len(self.ext)+1)] for x in gt_names]
            intersect = list(set(file_names) & set(gt_names))
            file_names = [x+'.jpg' for x in intersect]
            
        image_ids = []
        for x in file_names:
            if x.endswith(".jpg"):
                image_ids.append(x[:-4])
        self.image_ids = []

        meta_path = "/home/qchugroup/sdmcvpr2025/code/UNICL-SAM/ckpts/{}_{}_icl.pth".format('train' if is_train else 'val', dataset_name)
        if not os.path.exists(meta_path):
            meta_info = defaultdict(list)
            for img_id in tqdm(image_ids) :
                anno_path = os.path.join(self.anno_root, '{}.{}'.format(img_id, self.ext))
                label = Image.open(anno_path)
                if self.ext != 'tif':
                    label = label.convert('L')
                uni_cids = np.unique(np.asarray(label))
                if not self.zero_start:
                    if uni_cids[0] == 0:
                        uni_cids = uni_cids[1:]
                    if len(uni_cids) == 0:
                        print(img_id)
                    if uni_cids[-1] == self.ignore_idx:
                        uni_cids = uni_cids[:-1]
                # if len(uni_cids) >= 1 and self.zero_start or len(uni_cids) > 1:
                if len(uni_cids) >= 1 :
                    self.image_ids.append(img_id)
                for cid in uni_cids:
                    if cid in self.class_ids:
                        meta_info[cid].append(img_id)
            self.meta_info = meta_info
            torch.save([self.meta_info, self.image_ids], meta_path)
        else :
            self.meta_info, self.image_ids = torch.load(meta_path)

    def get_meta(self, idx):
        cid = self.class_ids[idx]
        ref_img_id = self._get_ref_cid(cid, None)
        if ref_img_id is not None:
            ref_id = self.image_ids.index(ref_img_id)
        else :
            ref_id = 3
        return ref_id        

        # outputs = []
        # for cid in self.get_class_ids():
        #     ref_img_id = self._get_ref_cid(cid, None)
        #     if ref_img_id is not None:
        #         ref_id = self.image_ids.index(ref_img_id)
        #     else :
        #         ref_id = 3
        #         self.image_ids[0]
        #     outputs.append(self.__getitem__(ref_id, [cid]))
        # return outputs

    def get_class_ids(self,):
        return np.array(sorted(list(self.all_cls)))

    def get_class_names(self,):
        cls_ids = self.get_class_ids()
        return [self.cid_to_cat[x] for x in cls_ids]

    def mask_transform(self, mask):
        if isinstance(mask, np.ndarray):
            if len(mask.shape) == 3:
                mask = Image.fromarray(np.squeeze(mask))
            else:
                mask = Image.fromarray(mask)
        return torch.LongTensor(np.array(self.output_transform(mask)).astype('int8'))

    def _get_info(self, img_id, cats_list=None, ret_uni_cids=None, sample_max_inst=True, supp=False):
        # print('img', img_id)
        image = Image.open(os.path.join(self.img_root, '{}.jpg'.format(img_id))).convert('RGB')
        masks = Image.open(os.path.join(self.anno_root, '{}.{}'.format(img_id, self.ext)))
        if self.ext != 'tif':
            masks = masks.convert('L')
        masks = np.array(masks)
        uni_cids = np.unique(masks)
        if not self.zero_start:
            if uni_cids[0] == 0:
                uni_cids = uni_cids[1:]
            if uni_cids[-1] == self.ignore_idx:
                uni_cids = uni_cids[:-1]

        if cats_list is None :
            if sample_max_inst and len(uni_cids) > self.num_samples :
                cats_list = np.random.choice(
                    uni_cids, size=self.num_samples, replace=False
                ).tolist()
            else :
                cats_list = uni_cids.tolist()
        else :
            uni_cids = np.array(cats_list)

        masks_list = []
        for cid in cats_list :
            masks_list.append(masks==cid)

        masks = np.stack(masks_list)
        def to_tensor(x):
            return torch.tensor(np.array(x), dtype=torch.float32).permute(2,0,1)

        if (self.transforms is not None) and (not supp):
            image = self.sam_transform.apply_image(np.array(image))
            image = torch.tensor(image).permute(2, 0, 1)
            image = self.output_transform(image).float()
            masks = self.mask_transform(masks).float().unsqueeze(0)

        if ret_uni_cids :
            return image, masks, cats_list, uni_cids
        return image, masks, cats_list

    def __len__(self,):
        if self.is_meta:
            return len(self.class_ids)
        return len(self.image_ids)

    def _get_ref_cid(self, cid, index):
        idx_list = self.meta_info[cid]
        ref_index = index
        
        if not len(idx_list):
            return None

        if len(idx_list) > 1 or ref_index is None:
            while ref_index == index:
                ref_index = random.choice(idx_list)
        # else:
        #     return self._get_ref_cid(1, None)
        return ref_index

    def __getitem__(self, index, cat_id=None):
        if self.is_meta :
            cat_id = [self.class_ids[index]]
            index = self.get_meta(index)

        img_id = self.image_ids[index]
        image, masks, cats_list, uni_cids = self._get_info(img_id, cat_id, ret_uni_cids=True)

        support_imgs, support_masks = [], []

        sample = {
            'image': image,
            'label': masks*255,
            'is_inst': False,
            # "img_id": torch.from_numpy(np.array(int(img_id))),
            "shape": torch.tensor(image.shape[-2:]),
            # 'info': self.dataset_name,
            # 'num_samples': torch.from_numpy(np.array(self.num_samples)),
        }

        if self.is_meta:
            sample.update(class_id=cat_id[0])

        if cat_id is None :
            for cid in cats_list:
                ref_img_id = self._get_ref_cid(cid, img_id)
                image_ref, masks_ref, _ = self._get_info(ref_img_id, [cid], supp=True)
                if self.transforms is not None:
                    image_ref, masks_ref = self.transforms(image_ref, Image.fromarray(masks_ref.squeeze()))
                support_imgs.append(image_ref)
                support_masks.append(masks_ref)

            # FIXME: only support num_inst == 1
            sample.update({
                'supp_image': torch.stack(support_imgs)[0]*255,
                'supp_label': torch.stack(support_masks)[0]*255,
            })

            # return super().__getitem__(index)
            # sample['neg_class_names'] = [self.cid_to_cat[cid] for cid in (self.all_cls - set(uni_cids.tolist()))]

        if self.is_semseg and not self.is_meta:
            _, masks, cat_ids = self._get_info(img_id, sample_max_inst=False)
            all_cats = np.array(self.get_class_ids())
            cat_ids = np.array(cat_ids) #(ninst, )
            # if not self.zero_start:
            #     cat_ids -= 1    
            # semmask = np.zeros((len(all_cats), *masks.shape[-2:]))
            # semmask[cat_ids] = masks
            all_cats_this = cat_ids[None] == all_cats[:, None]
 
            # ninst, h, w = masks.shape
            # semmask = np.zeros((len(all_cats), *masks.shape[-2:]))
            # semmask[all_cats_this.sum(-1)>0] = masks.numpy()
            semmask = masks
            sample['origin_semmask'] = semmask
            sample['valid_cids'] = all_cats_this.sum(-1)>0

        if False :
            import cv2
            cv2.imwrite('tmp.jpg', image.permute(1,2,0).int().numpy())
            cv2.imwrite('tmp.jpg', sample['image'].permute(1,2,0).int().numpy())
            cv2.imwrite('tmp.jpg', sample['image_dual'].permute(1,2,0).int().numpy())
            pass

        return sample

class SAMSegADEContrastiveDataset(VisionDataset):
    def __init__(self, 
                 root,
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 is_train=True, 
                 dataset_name='ade20k', 
                 size=128, 
                 num_samples=1,
                 is_semseg=False, 
                 ext='png', 
                 is_meta=False,
                 label_select_type=None):
        super().__init__(root, transforms, transform, target_transform)

        self.img_size = size
        self.num_samples = num_samples
        self.ext = ext
        self.split = split = 'training' if is_train else 'validation'
        self.is_semseg = is_semseg
        self.zero_start = False
        self.is_meta = is_meta
        self.dataset_name = dataset_name
        self.label_select_type = label_select_type
        self.sam_transform = ResizeLongestSide(self.img_size)
        self.output_transform = T.Resize((self.img_size, self.img_size), antialias=True)
        classes_name_list = ADE_SEM_CLASSES
        if dataset_name in ('ade20k'):
            self.ignore_idx = 255
            self.img_root = os.path.join(root, "images", split)
            self.anno_root = os.path.join(root, "annotations", split)
            classes_name_list = ['none'] + classes_name_list
        elif dataset_name in ('sd_ade20k'):
            self.ignore_idx = 255
            self.img_root = os.path.join(root, "images_detectron2", split)
            self.anno_root = os.path.join(root, "annotations_detectron2", split)
            classes_name_list = ['none'] + classes_name_list
        elif dataset_name == 'cocostuff':
            self.img_root = os.path.join(root, "images", split)
            self.anno_root = os.path.join(root, "annotations", split)
            classes_name_list = [x.split(':')[-1] for x in classes_name_list]
        elif dataset_name == 'ade847':
            self.ignore_idx = 65535
            self.img_root = os.path.join(root, "images_detectron2", split)
            self.anno_root = os.path.join(root, "annotations_detectron2", split)
            classes_name_list = ['none'] + classes_name_list
        elif dataset_name == 'sd_ade847':
            self.zero_start = False
            self.ignore_idx = 65535
            self.img_root = os.path.join(root, "images_detectron2", split)
            self.anno_root = os.path.join(root, "annotations_detectron2", split)
            classes_name_list = ['none'] + classes_name_list
        elif dataset_name == 'pc459':
            assert not is_train
            self.zero_start = True
            self.split = split = 'training' if is_train else 'validation'
            self.img_root = os.path.join(root, 'JPEGImages')
            self.anno_root = os.path.join(root, 'annotations_detectron2/pc459_val')
        else :
            raise NotImplementedError

        self.cid_to_cat = np.array(classes_name_list)
        self.all_cls = set(list(range(len(self.cid_to_cat)))) 
        if not self.zero_start:
            self.all_cls = self.all_cls- {0}
        self.class_ids = self.get_class_ids()
        file_names = sorted(
            os.listdir(self.img_root)
        )
        
        gt_names = os.listdir(self.anno_root)
        if len(file_names) != len(gt_names):
            print('warning, not equal')
            file_names = [x[:-len('.jpg')] for x in file_names]
            gt_names = [x[:-(len(self.ext)+1)] for x in gt_names]
            intersect = list(set(file_names) & set(gt_names))
            file_names = [x+'.jpg' for x in intersect]
            
        image_ids = []
        for x in file_names:
            if x.endswith(".jpg"):
                image_ids.append(x[:-4])
        self.image_ids = []

        meta_path = "/home/qchugroup/sdmcvpr2025/code/try/SEGIC/utils/dataset/{}_{}_icl.pth".format('train' if is_train else 'val', dataset_name)
        if not os.path.exists(meta_path):
            meta_info = defaultdict(list)
            for img_id in tqdm(image_ids) :
                anno_path = os.path.join(self.anno_root, '{}.{}'.format(img_id, self.ext))
                label = Image.open(anno_path)
                if self.ext != 'tif':
                    label = label.convert('L')
                uni_cids = np.unique(np.asarray(label))
                if not self.zero_start:
                    if uni_cids[0] == 0:
                        uni_cids = uni_cids[1:]
                    if len(uni_cids) == 0:
                        print(img_id)
                    if uni_cids[-1] == self.ignore_idx:
                        uni_cids = uni_cids[:-1]
                # if len(uni_cids) >= 1 and self.zero_start or len(uni_cids) > 1:
                if len(uni_cids) >= 1 :
                    self.image_ids.append(img_id)
                for cid in uni_cids:
                    if cid in self.class_ids:
                        meta_info[cid].append(img_id)
            self.meta_info = meta_info
            torch.save([self.meta_info, self.image_ids], meta_path)
        else :
            self.meta_info, self.image_ids = torch.load(meta_path)

    def get_meta(self, idx):
        cid = self.class_ids[idx]
        ref_img_id = self._get_ref_cid(cid, None)
        if ref_img_id is not None:
            ref_id = self.image_ids.index(ref_img_id)
        else :
            ref_id = 3
        return ref_id        

        # outputs = []
        # for cid in self.get_class_ids():
        #     ref_img_id = self._get_ref_cid(cid, None)
        #     if ref_img_id is not None:
        #         ref_id = self.image_ids.index(ref_img_id)
        #     else :
        #         ref_id = 3
        #         self.image_ids[0]
        #     outputs.append(self.__getitem__(ref_id, [cid]))
        # return outputs

    def get_class_ids(self,):
        return np.array(sorted(list(self.all_cls)))

    def get_class_names(self,):
        cls_ids = self.get_class_ids()
        return [self.cid_to_cat[x] for x in cls_ids]

    def mask_transform(self, mask):
        if isinstance(mask, np.ndarray):
            if len(mask.shape) == 3:
                mask = Image.fromarray(np.squeeze(mask))
            else:
                mask = Image.fromarray(mask)
        return torch.LongTensor(np.array(self.output_transform(mask)).astype('int8'))

    def _get_info(self, img_id, cats_list=None, ret_uni_cids=None, sample_max_inst=True, supp=False):
        # print('img', img_id)
        image = Image.open(os.path.join(self.img_root, '{}.jpg'.format(img_id))).convert('RGB')
        masks = Image.open(os.path.join(self.anno_root, '{}.{}'.format(img_id, self.ext)))
        if self.ext != 'tif':
            masks = masks.convert('L')
        masks = np.array(masks)
        uni_cids = np.unique(masks)
        if not self.zero_start:
            if uni_cids[0] == 0:
                uni_cids = uni_cids[1:]
            if uni_cids[-1] == self.ignore_idx:
                uni_cids = uni_cids[:-1]

        if cats_list is None :
            if sample_max_inst and len(uni_cids) > self.num_samples :
                cats_list = np.random.choice(
                    uni_cids, size=self.num_samples, replace=False
                ).tolist()
            else :
                cats_list = uni_cids.tolist()
        else :
            uni_cids = np.array(cats_list)

        masks_list = []
        for cid in cats_list :
            masks_list.append(masks==cid)

        masks = np.stack(masks_list)

        if (self.transforms is not None) and (not supp):
            image = self.sam_transform.apply_image(np.array(image))
            image = torch.tensor(image).permute(2, 0, 1)
            image = self.output_transform(image)
            masks = self.mask_transform(masks).unsqueeze(0)
        else:
            image = np.array(image)
            masks = masks[0].astype(image.dtype)

        if ret_uni_cids :
            return image, masks, cats_list, uni_cids
        return image, masks, cats_list

    def __len__(self,):
        if self.is_meta:
            return len(self.class_ids)
        return len(self.image_ids)

    def _get_ref_cid(self, cid, index):
        idx_list = self.meta_info[cid]
        ref_index = index
        
        if not len(idx_list):
            return None

        if len(idx_list) > 1 or ref_index is None:
            while ref_index == index:
                ref_index = random.choice(idx_list)
        # else:
        #     return self._get_ref_cid(1, None)
        return ref_index

    def __getitem__(self, index, cat_id=None):
        if self.is_meta :
            cat_id = [self.class_ids[index]]
            index = self.get_meta(index)

        img_id = self.image_ids[index]
        image, masks, cats_list, uni_cids = self._get_info(img_id, cat_id, ret_uni_cids=True)

        support_imgs, support_masks = [], []
        support_ctr_imgs, support_ctr_masks = [], []

        sample = {
            'image': image,
            'label': masks,
            'is_inst': False,
            'info': self.dataset_name,
            'num_samples': torch.from_numpy(np.array(self.num_samples)),
        }

        if self.is_meta:
            sample.update(class_id=cat_id[0])

        if self.label_select_type == None:
            label_select_type = random.choice(["bbox", "dilate", "erode", "None"])
        else:
            label_select_type = self.label_select_type

        if cat_id is None :
            for cid in cats_list:
                ref_img_id = self._get_ref_cid(cid, img_id)
                image_ref, masks_ref, _ = self._get_info(ref_img_id, [cid], supp=True)
                image_ref_ctr = deepcopy(image_ref) 
                masks_ref_ct = deepcopy(masks_ref)

                for i in range(3):
                    try:
                        temp_img, temp_mask = apply_transform(image_ref_ctr, masks_ref_ct)
                        temp_mask = (temp_mask > 0).astype(temp_img.dtype)
                        assert temp_mask.max() > 0
                        image_ref_ctr, masks_ref_ct = temp_img, temp_mask
                        break
                    except:
                        pass

                if self.transforms is not None:
                    image_ref, masks_ref = self.transforms(Image.fromarray(image_ref), Image.fromarray(masks_ref))
                    image_ref_ctr, masks_ref_ct = self.transforms(Image.fromarray(image_ref_ctr), Image.fromarray(masks_ref_ct))

                if label_select_type == 'dilate':
                    masks_ref_ct = dilate(masks_ref_ct, ksize=10)
                    masks_ref_ct = self.output_transform(masks_ref_ct)
                elif label_select_type == 'erode':
                    masks_ref_ct = erode(masks_ref_ct, ksize=10)                
                    masks_ref_ct = self.output_transform(masks_ref_ct)
                elif label_select_type == 'bbox':
                    try:
                        idx = torch.where(masks_ref_ct > 0)
                        y_min, y_max = torch.min(idx[1], dim=0)[0].item(), torch.max(idx[1], dim=0)[0].item()
                        x_min, x_max = torch.min(idx[2], dim=0)[0].item(), torch.max(idx[2], dim=0)[0].item()

                        masks_ref_ct = torch.zeros_like(masks_ref_ct)
                        masks_ref_ct[:, y_min:y_max, x_min:x_max] = 1  
                    except:
                        pass
                else:
                    pass   

                support_imgs.append(image_ref)
                support_masks.append(masks_ref)
                support_ctr_imgs.append(image_ref_ctr)
                support_ctr_masks.append(masks_ref_ct)

            # FIXME: only support num_inst == 1
            sample.update({
                'supp_image': torch.stack(support_imgs),
                'supp_label': torch.stack(support_masks),
                'supp_ctr_image': torch.stack(support_ctr_imgs),
                'supp_ctr_label': torch.stack(support_ctr_masks)
            })

            # return super().__getitem__(index)
            # sample['neg_class_names'] = [self.cid_to_cat[cid] for cid in (self.all_cls - set(uni_cids.tolist()))]

        if self.is_semseg and not self.is_meta:
            _, masks, cat_ids = self._get_info(img_id, sample_max_inst=False)
            all_cats = np.array(self.get_class_ids())
            cat_ids = np.array(cat_ids) #(ninst, )
            # if not self.zero_start:
            #     cat_ids -= 1    
            # semmask = np.zeros((len(all_cats), *masks.shape[-2:]))
            # semmask[cat_ids] = masks
            all_cats_this = cat_ids[None] == all_cats[:, None]
 
            # ninst, h, w = masks.shape
            # semmask = np.zeros((len(all_cats), *masks.shape[-2:]))
            # semmask[all_cats_this.sum(-1)>0] = masks.numpy()
            semmask = masks
            sample['origin_semmask'] = semmask
            sample['valid_cids'] = all_cats_this.sum(-1)>0

        if False :
            import cv2
            cv2.imwrite('tmp.jpg', image.permute(1,2,0).int().numpy())
            cv2.imwrite('tmp.jpg', sample['image'].permute(1,2,0).int().numpy())
            cv2.imwrite('tmp.jpg', sample['image_dual'].permute(1,2,0).int().numpy())
            pass

        return sample

class MixSemSegContrastiveDataset(ConcatDataset):
    """Wrapper for mix segmentation and unseg data"""

    def __init__(self, transform, target_transform, size=518, num_samples=1, stage='train'): # , tokenizer
        if stage=='train':
            super().__init__([SAMImgSegContrastiveDataset(COCO_ROOT_TRAIN, transform=transform, target_transform=target_transform, size=size, is_train=True, dataset_name='coco_train'), 
                              SAMSegADEContrastiveDataset(ADE_ROOT, transform=transform, target_transform=target_transform, size=size, num_samples=num_samples, is_train=True),
                              SAMSegLVISContrastiveDataset(LVIS_ROOT, transform=transform, target_transform=target_transform, size=size, num_samples=num_samples, is_train=True),
                            ])
        else:
            super().__init__([SAMImgSegContrastiveDataset(COCO_ROOT_VAL, transform=transform, target_transform=target_transform, size=size, is_train=False, dataset_name='coco_val'), 
                              SAMSegADEContrastiveDataset(ADE_ROOT, transform=transform, target_transform=target_transform, size=size, num_samples=num_samples, is_train=False)])
        self.num_samples = num_samples

        self.dataset_length = np.array([len(ds) for ds in self.datasets])
        self.dataset_scale = np.array([0] + np.cumsum(self.dataset_length).tolist()[:-1])
        self.dataset_names = [ds.__class__.__name__ for ds in self.datasets]

        base = np.sum(np.array([np.power(l, 2) for l in self.dataset_length]))
        self.sampling_weights = {name: np.power(self.dataset_length[i],2) / base for i, name in enumerate(self.dataset_names)}

        del base

    @property
    def coco(self):
        return self.datasets[0]

    @property
    def ade(self):
        return self.datasets[1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            
        return self.datasets[dataset_idx][sample_idx]

class MixSemSegCOCODataset(ConcatDataset):
    """Wrapper for mix segmentation and unseg data"""

    def __init__(self, transform, target_transform, size=518, num_samples=1, stage='train'): # , tokenizer
        super().__init__([FSSCOCODataset(datapath='/home/qchugroup/sdmcvpr2025/datasets/coco', fold=0, split=stage, shot=1, transform=transform, target_transform=target_transform),
                          FSSCOCODataset(datapath='/home/qchugroup/sdmcvpr2025/datasets/coco', fold=1, split=stage, shot=1, transform=transform, target_transform=target_transform), 
                          FSSCOCODataset(datapath='/home/qchugroup/sdmcvpr2025/datasets/coco', fold=2, split=stage, shot=1, transform=transform, target_transform=target_transform),
                          FSSCOCODataset(datapath='/home/qchugroup/sdmcvpr2025/datasets/coco', fold=3, split=stage, shot=1, transform=transform, target_transform=target_transform)])
        self.num_samples = num_samples

        self.dataset_length = np.array([len(ds) for ds in self.datasets])
        self.dataset_scale = np.array([0] + np.cumsum(self.dataset_length).tolist()[:-1])
        self.dataset_names = [ds.__class__.__name__ for ds in self.datasets]

        base = np.sum(np.array([np.power(l, 2) for l in self.dataset_length]))
        self.sampling_weights = {name: np.power(self.dataset_length[i],2) / base for i, name in enumerate(self.dataset_names)}

        del base

    @property
    def coco(self):
        return self.datasets[0]

    @property
    def ade(self):
        return self.datasets[1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            
        return self.datasets[dataset_idx][sample_idx]

class CustomConcatDataset(Dataset):
    def __init__(self, dataset_list, dataset_ratio=None, samples_per_epoch=160000):
        self.dataset_list = dataset_list
        if dataset_ratio is not None:
            assert len(dataset_ratio) == len(dataset_list)
        else :
            dataset_ratio = [1] * len(dataset_list)
        self.dataset_ratio = dataset_ratio
        if samples_per_epoch is not None:
            self.samples_per_epoch = samples_per_epoch
        else:
            self.samples_per_epoch = sum([len(ds) for ds in dataset_list])

    def __len__(self,):
        return self.samples_per_epoch

    def __getitem__(self, index):
        dataset_idx = random.choices(list(range(len(self.dataset_ratio))), weights=self.dataset_ratio, k=1)[0]
        dataset = self.dataset_list[dataset_idx]
        index = random.randint(0, len(dataset) - 1)
        return dataset[index]

class GetOutOfLoop(Exception):
    pass


'''
few-shot segmentation
'''
def find_bbox(mask):
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return stats[1:]  # remove bg stat

def transform_anns(mask, ann_type):
    mask_ori = mask.copy()

    if ann_type == 'bbox':
        bboxs = find_bbox(mask)
        for j in bboxs: 
            cv2.rectangle(mask, (j[0], j[1]), (j[0] + j[2], j[1] + j[3]), 1, -1) # -1->fill; 2->draw_rec        
        return mask, mask_ori
    
    elif ann_type == 'mask':
        return mask, mask_ori

def load_fss_coco_seg_json(json_file, image_root, img_valid_ids, class_lists, dataset_name=None, extra_annotation_keys=None):
    from pycocotools.coco import COCO
    from fvcore.common.timer import Timer
    from detectron2.utils.file_io import PathManager
    import contextlib
    import io
    
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    # sort indices for reproducible results
    # img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_valid_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_valid_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = {}
    coco_valid_id_name_map = coco_id_name_map[class_lists]
    per_cat_pool={int(i): [] for i, cat in coco_valid_id_name_map.items()}

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    record_id = 0 # id for each anno
    for (img_dict, anno_dict_list) in imgs_anns:
        if min(img_dict["height"], img_dict["width"]) < 128:
            continue

        for cat_id, cat in coco_valid_id_name_map.items(): 
            cat_anns = [x for index, x in enumerate(anno_dict_list) if 
            {k: v for k, v in x.items() if (k == "category_id" and v == cat_id)}
            ]
            if len(cat_anns) > 0:
                record = {}
                record["file_name"] = os.path.join(image_root, img_dict["file_name"])
                record["height"] = img_dict["height"]
                record["width"] = img_dict["width"]
                image_id = record["image_id"] = img_dict["id"]
                objs = []
                for anno in cat_anns:
                    assert anno["image_id"] == image_id

                    assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

                    obj = {key: anno[key] for key in ann_keys if key in anno}
                    if "bbox" in obj and len(obj["bbox"]) == 0:
                        raise ValueError(
                            f"One annotation of image {image_id} contains empty 'bbox' value! "
                            "This json does not have valid COCO format."
                        )

                    segm = anno.get("segmentation", None)
                    if segm:  # either list[list[float]] or dict(RLE)
                        if isinstance(segm, dict):
                            if isinstance(segm["counts"], list):
                                # convert to compressed RLE
                                segm = mask_util.frPyObjects(segm, *segm["size"])
                        else:
                            # filter out invalid polygons (< 3 points)
                            segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                            if len(segm) == 0:
                                num_instances_without_valid_segmentation += 1
                                continue  # ignore this instance
                            segm = mask_util.frPyObjects(segm, img_dict["height"], img_dict["width"])
                            segm = mask_util.merge(segm)
                        obj["segmentation"] = segm

                    keypts = anno.get("keypoints", None)
                    if keypts:  # list[int]
                        for idx, v in enumerate(keypts):
                            if idx % 3 != 2:
                                keypts[idx] = v + 0.5
                        obj["keypoints"] = keypts

                    if id_map:
                        annotation_category_id = obj["category_id"]
                        try:
                            obj["category_id"] = id_map[annotation_category_id]
                        except KeyError as e:
                            raise KeyError(
                                f"Encountered category_id={annotation_category_id} "
                                "but this id does not exist in 'categories' of the json file."
                            ) from e
                    objs.append(obj)
                if len(objs) == 0:
                    continue
                record["annotations"] = objs
                record["category_id"] = cat_id
                per_cat_pool[cat_id].append(record_id)
                dataset_dicts[record_id] = record
                record_id += 1

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts, per_cat_pool

class FSSCOCODataset(VisionDataset):
    img_size=518

    def __init__(self, datapath, fold, split, shot, transform=None, target_transform=None, transforms=None, use_original_imgsize=False):
        super().__init__(datapath, transforms, transform, target_transform)
        
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.split_coco = split if split == 'val2014' else 'train2014'
        self.base_path = datapath
        self.use_original_imgsize = use_original_imgsize

        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

        self.sam_transform = ResizeLongestSide(self.img_size)
        self.output_transform = T.Resize((self.img_size, self.img_size), antialias=True)

    def __len__(self):
        return len(self.img_metadata) if self.split in ['trn', 'test'] else 1000
        # return len(self.img_metadata)

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()

        query_img = self.output_transform(query_img)
        query_mask = query_mask.float()
        if not self.use_original_imgsize:
            query_mask = interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size[-2:], mode='nearest').squeeze(0)

        # import pdb; pdb.set_trace()
        if self.transforms is not None:
            query_img = self.sam_transform.apply_image(np.array(query_img))
            query_img = torch.tensor(query_img).permute(2, 0, 1)

        support_imgs = [self.transform(support_img) for support_img in support_imgs]
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs[0].shape[-2:], mode='nearest').squeeze(0)

        sample = {
            'image': query_img,
            'label': query_mask,
            'supp_image': torch.stack(support_imgs),
            'supp_label': torch.stack(support_masks),
            'supp_ctr_image': torch.stack(support_imgs),
            'supp_ctr_label': torch.stack(support_masks),
            'is_inst': False,
            'name': 'COCO-20i',
            'num_samples': torch.from_numpy(np.array(self.shot)),
            'fold': torch.from_numpy(np.array(self.fold))
        }

        return sample

        refer = {}
        refer['input'] = query_img
        refer['output'] = query_mask

        samples = {}
        samples['input'] = support_imgs
        samples['output'] = support_masks

        return dict(samples=samples,
                    refer=refer,
                    name='FSS',
                    num_samples=self.shot,
                    class_id=class_sample
        )

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val

        return class_ids

    def build_img_metadata_classwise(self):
        import pickle
        with open('/home/qchugroup/sdmcvpr2025/datasets/coco/fss_data_lists/split/coco/%s/fold%d.pkl' % (self.split, self.fold), 'rb') as f:
            img_metadata_classwise = pickle.load(f)
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = torch.tensor(np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png')))
        return mask

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        query_mask = self.read_mask(query_name)

        org_qry_imsize = query_img.size

        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []
        for support_name in support_names:
            support_imgs.append(Image.open(os.path.join(self.base_path, support_name)).convert('RGB'))
            support_mask = self.read_mask(support_name)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize

'''
FSS-1000
'''
class FSS1000Dataset(VisionDataset):
    def __init__(self, base_data_root, transform, target_transform=None, transforms=None, split='test', shot=1, img_size=518):
        super().__init__(base_data_root, transforms, transform, target_transform)
        self.split = split
        self.benchmark = 'fss1000'
        self.shot = shot
        self.img_size = img_size

        self.base_path = base_data_root

        # Given predefined test split, load randomly generated training/val splits:
        # (reference regarding trn/val/test splits: https://github.com/HKUSTCV/FSS-1000/issues/7))
        with open(f'{self.base_path}/%s.txt' % split, 'r') as f:
            self.categories = f.read().split('\n')[:-1]
        self.categories = sorted(self.categories)

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()

        self.output_transform = T.Resize((self.img_size, self.img_size), antialias=True)

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        # if True:
        #     # import time
        #     # random.seed(int(time.time()))
        #     idx_x = random.choice(list(range(len(self))))
        #     _, support_names, _ = self.sample_episode(idx_x)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        def to_tensor(x):
            return torch.tensor(np.array(x), dtype=torch.float32).permute(2,0,1)
        assert len(support_imgs) == 1 and len(support_masks) == 1
        support_imgs, support_masks = support_imgs[0], support_masks[0]
        query_img, support_imgs = [to_tensor(x) for x in [query_img, support_imgs]]
        query_mask, support_masks = query_mask[None].float(), support_masks[None].float()
        if query_img.shape[-2:] != query_mask.shape[-2:] or support_imgs[0].shape[-2:] != support_masks[0].shape[-2:]:
            # bugs caused by mismatch of gt and img
            return self.__getitem__(idx+1)

        query_img, query_mask = self.output_transform(query_img), self.output_transform(query_mask)
        query_mask = (query_mask > 0).float()
        support_imgs, support_masks = self.output_transform(support_imgs), self.output_transform(support_masks)
        support_masks = (support_masks > 0).float()

        sample = {
            'image': query_img,
            'label': query_mask,
            'supp_image': torch.stack([support_imgs])/255,
            'supp_label': torch.stack([support_masks]),
            'is_inst': False,
            'info': 'FSS-1000',
            'num_samples': torch.from_numpy(np.array(self.shot)),
        }

        return sample

    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_id = query_name.split('/')[-1].split('.')[0]
        query_name = os.path.join(os.path.dirname(query_name), query_id) + '.png'
        support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
        support_names = [os.path.join(os.path.dirname(name), sid) + '.png' for name, sid in zip(support_names, support_ids)]

        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        class_sample = self.categories.index(query_name.split('/')[-2])
        if self.split == 'val':
            class_sample += 520
        elif self.split == 'test':
            class_sample += 760

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(range(1, 11), 1, replace=False)[0]
            support_name = os.path.join(os.path.dirname(query_name), str(support_name)) + '.jpg'
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample

    def build_class_ids(self):
        if self.split == 'trn':
            class_ids = range(0, 520)
        elif self.split == 'val':
            class_ids = range(520, 760)
        elif self.split == 'test':
            class_ids = range(760, 1000)
        return class_ids

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            img_paths = sorted([path for path in glob('%s/*' % os.path.join(self.base_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata.append(img_path)
        return img_metadata

def get_mask_area(image_feature, mask):
    feat = image_feature.flatten(1).permute(1,0)
    mask = mask.flatten()
    mask_feat = feat[mask>0]

    if len(mask_feat) == 0:
        mask_feat = feat

    return mask_feat

if __name__=='__main__':

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

    # seg_dataset = SAMSegUnoverlapDataset(COCO_ROOT_TRAIN, sam_seg_json, transform=image_transform, target_transform=mask_transform, num_samples=1, size=512)
    # ['gaussian', 'gray', 'color_jitter', 'sharp', 'horizontal_flip', 'vertical_flip', 
    #                             'posterize', 'solarize', 'equalize', 'jpeg_compression', 'gaussian_noise', 
    #                             'motion_blur', 'cartoon', 'mean_shift_blur', 'light']
    # seg_dataset = SAMSegLVISDataset(LVIS_ROOT, 
    #                                 transform=image_transform, 
    #                                 target_transform=mask_transform, 
    #                                 num_samples=1, 
    #                                 is_train=False, 
    #                                 fold=0,
    #                                 size=512) # 'gaussian','gray','color_jitter','sobel','canny','sharp','jpeg_compression','gaussian_noise','motion_blur'
    # seg_dataset = SAMSegADEDataset(ADE_ROOT, 
    #                                 transform=image_transform, 
    #                                 target_transform=mask_transform, 
    #                                 num_samples=1, 
    #                                 is_train=True, 
    #                                 size=518) 
    seg_dataset = CustomConcatDataset([
            SAMImgSegDataset(transform=image_transform, target_transform=mask_transform, size=518, is_train=True, dataset_name='coco_train'), 
            SAMSegADEDataset(ADE_ROOT, transform=image_transform, target_transform=mask_transform, size=518, num_samples=1, is_train=True),
            SAMSegLVISDataset(LVIS_ROOT, transform=image_transform, target_transform=mask_transform, size=518, num_samples=1, is_train=True)],
            None, samples_per_epoch=None)
    # seg_dataset = MySegDataset(seg_image_root, seg_json_file, transform=image_transform, target_transform=mask_transform)
    # seg_dataset = SAMSegLVISContrastiveDataset(LVIS_ROOT, 
    #                                 transform=image_transform, 
    #                                 target_transform=mask_transform, 
    #                                 num_samples=1,
    #                                 is_train=False, 
    #                                 size=518)
    # seg_dataset = MixSemSegContrastiveDataset(transform=image_transform, target_transform=mask_transform, num_samples=1, size=518, stage='train')

    print(seg_dataset)
    data_loader = torch.utils.data.DataLoader(
        seg_dataset,
        batch_size=4,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )

    # # from pathlib import Path
    # # from torchvision.utils import save_image
    # # Path('analysis/test').mkdir(parents=True, exist_ok=True)
    # # transform = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST)
    for num, data in tqdm(enumerate(data_loader)):
        print(data['is_inst'])
    #     samples, refers = data['samples'], data['refer']
    #     sample_refers, sample_gts, refer, gt = samples['input'], samples['output'], refers['input'], refers['output']
    #     B, C, H, W = refer.shape

    #     sample_imgs = []
    #     for i in range(len(sample_refers)):
    #         sample_refer = sample_refers[i]
    #         sample_gt = sample_gts[i]
    #         sample_pair = torch.cat((torch.clip(transform(sample_refer) *255, 0, 255),torch.clip(transform(sample_gt) *255, 0, 255)),-1)
    #         sample_imgs.append(sample_pair)
        
    #     sample_imgs.append(torch.cat((torch.clip(transform(refer), 0, 255), torch.clip(transform(gt).repeat(1, 3, 1, 1) *255, 0, 255)),-1)) # .repeat(1, 3, 1, 1)
    #     gt_canvas = torch.cat(sample_imgs, dim=-1)
    #     save_image(gt_canvas/255, f'analysis/test/{num}.png', nrow=1)  
    # print(seg_dataset)

    '''
    FSS
    '''
    # dataset_val = FSS1000Dataset(base_data_root='/home/qchugroup/sdmcvpr2025/datasets/fss-1000', 
    #                          transform=image_transform, target_transform=mask_transform, 
    # )

    # data_loader = torch.utils.data.DataLoader(
    #     dataset_val,
    #     batch_size=4,
    #     num_workers=0,
    #     pin_memory=True,
    #     drop_last=True,
    #     shuffle=True
    # )

    # for num, data in tqdm(enumerate(data_loader)):
    #     print(data['info'])

