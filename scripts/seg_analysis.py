from test import *
import clip
from torchvision.datasets.vision import VisionDataset
import pickle
import torch.nn.functional as F
import torchvision.transforms as T
import torch
from unicl_sam.data import value2key, dilate, erode, Sobel, Canny

class SAMSegSimDataset(VisionDataset):
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
        clip_features=None,
        cat='person',
        select_type=None
    ):
        super().__init__(root, transforms, transform, target_transform)

        with open(annFile,'r') as fr: 
            seg_json = json.load(fr) 

        self.ds, self.per_cat_pool = np.array(seg_json['dataset_dicts']), np.array(seg_json['per_cat_pool'])
        self.size = size
        self.num_samples = num_samples

        self.cat = cat
        self.select_type = select_type
        self.clip_features = clip_features

        self.scale = (0.1, 1)
        self.ratio = (3.0/4.0, 4.0/3.0)
        self.crop = RandomResizedCrop(self.size, scale=self.scale, ratio=self.ratio)
        self.sam_transform = ResizeLongestSide(self.img_size)
        self.output_transform = T.Resize((self.img_size, self.img_size))

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

    def cos_similarity(self, vec1, vec2, cat, num):
        if len(vec2.shape) == 1:
            vec2 = vec2.unsqueeze(0)

        cos_sim = F.cosine_similarity(vec1, vec2, dim=1)

        if len(cos_sim) > num:
            max_id = torch.topk(cos_sim, k=num+1).indices
            max_id = max_id[1:]
            min_id = torch.topk(cos_sim, k=num+1, largest=False).indices
            min_id = min_id[1:]
        else:
            max_id = torch.multinomial(cos_sim, num_samples=num, replacement=True)
            min_id = torch.multinomial(cos_sim, num_samples=num, replacement=True)

        max_id += self.per_cat_pool[()][str(cat)][0]
        min_id += self.per_cat_pool[()][str(cat)][0]

        return max_id, min_id

    def mix_cos_similarity(self, vec1, vec2, cat, num, select):
        if len(vec2.shape) == 1:
            vec2 = vec2.unsqueeze(0)

        cos_sim = F.cosine_similarity(vec1, vec2, dim=1)

        # import pdb; pdb.set_trace()

        select = select.split('-')[1:]

        if len(cos_sim) > num:
            max_id = torch.topk(cos_sim, k=int(select[0])+1).indices
            max_id = max_id[1:]
            min_id = torch.topk(cos_sim, k=int(select[1])+1, largest=False).indices
            min_id = min_id[1:]
        else:
            max_id = torch.multinomial(cos_sim, num_samples=int(select[0]), replacement=True)
            min_id = torch.multinomial(cos_sim, num_samples=int(select[1]), replacement=True)

        max_id += self.per_cat_pool[()][str(cat)][0]
        min_id += self.per_cat_pool[()][str(cat)][0]

        return torch.cat([min_id, max_id])

    def get_refer_record(self, cat_id, img_id):
        try:
            ids = random.sample(self.per_cat_pool[()][str(cat_id)], k=self.num_samples+1)
        except:
            ids = random.choices(self.per_cat_pool[()][str(cat_id)], k=self.num_samples+1)

        sample_records = [self.ds[()][str(id)] for id in ids]
        img_ids = [re['image_id'] for re in sample_records]

        if img_id in img_ids:
            idx = img_ids.index(img_id)
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

        refer = {}
        refer['input'] = refer_crop_img
        refer['output'] = refer_crop_mask
        refer['img_id'] = refer_img_id

        samples = {}
        samples['input'] = []
        samples['output'] = []
        samples['img_id'] = []

        vec1_img = self.clip_features[index].unsqueeze(0)
        vec2_img = self.clip_features[self.per_cat_pool[()][str(refer_cat_id)]] # [self.per_cat_pool[str(refer_synsets[0])]]

        if self.select_type == 'best':
            ids, _ = self.cos_similarity(vec1_img, vec2_img, refer_cat_id, self.num_samples)
        elif self.select_type == 'worst':
            _, ids = self.cos_similarity(vec1_img, vec2_img, refer_cat_id, self.num_samples)
        elif self.select_type.split('-')[0] == 'middle':
            ids = self.mix_cos_similarity(vec1_img, vec2_img, refer_cat_id, self.num_samples, self.select_type)
        else:
            raise ValueError()

        # import pdb; pdb.set_trace()
        for i in range(self.num_samples):
            sample_record = self.ds[()][str(ids[i].item())]
            sample_img_id = sample_record['image_id']
            sample_img_path = sample_record["file_name"]
            sample_img_shape = [sample_record["width"], sample_record["height"]]
            sample_segs = [ann["segmentation"] for ann in sample_record["annotations"]]
            sample_bboxs = [ann["bbox"] for ann in sample_record["annotations"]]

            sample_img = Image.open(sample_img_path).convert("RGB")
            sample_crop_img, sample_crop_mask = self.get_crop(np.array(sample_img), sample_img_shape, sample_bboxs, sample_segs, refer_cat_id)

            if self.transforms is not None:
                sample_crop_img, sample_crop_mask = self.transforms(sample_crop_img, sample_crop_mask)
            
            samples['input'].append(sample_crop_img)
            samples['output'].append(sample_crop_mask)
            samples['img_id'].append(sample_img_id)

        return dict(samples=samples,
                    refer=refer,
                    name='COCO_INC_S',
                    num_samples=self.num_samples
                    )

    def __len__(self):
        return len(self.ds[()])

class SAMSegDegradDataset(VisionDataset):
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
        cat='refer',
        select_type=None,
        **kwargs
    ):
        super().__init__(root, transforms, transform, target_transform)

        with open(annFile,'r') as fr: 
            seg_json = json.load(fr) 

        self.ds, self.per_cat_pool = np.array(seg_json['dataset_dicts']), np.array(seg_json['per_cat_pool'])
        self.size = size
        self.num_samples = num_samples

        self.cat = cat
        self.select_type = select_type

        self.scale = (0.1, 0.8)
        self.ratio = (3.0/4.0, 4.0/3.0)
        self.sam_transform = ResizeLongestSide(self.img_size)
        self.output_transform = T.Resize((self.img_size, self.img_size))

        # self.degradation_pools = {
        #     'gaussian': T.GaussianBlur(kernel_size=(kwargs['kernel'], kwargs['kernel']), sigma=50), # k=3,7,11,15
        #     'gray': T.Grayscale(3),
        #     'color_jitter': T.ColorJitter(brightness=(kwargs['bright'],kwargs['bright']), contrast=(kwargs['contrast'],kwargs['contrast']), saturation=(kwargs['saturation'],kwargs['saturation']), hue=kwargs['hue']), # b=0.5,1.5; c=0.5,1.5; s=0.5,1.5; h=0.1,0.3,0.5
        #     'sharp': T.RandomAdjustSharpness(kwargs['sharp'], p=1), # 5,10,15
        #     'horizontal_flip': T.RandomHorizontalFlip(p=1), 
        #     'vertical_flip': T.RandomVerticalFlip(p=1), 
        #     'posterize': T.RandomPosterize(bits=kwargs['bit'], p=1), # 1,2,3
        #     'solarize': T.RandomSolarize(threshold=kwargs['threshold'], p=1), # 0, 64, 128, 192, 256
        #     'invert': T.RandomInvert(p=1), # same to solarize t=0
        #     'equalize': T.RandomEqualize(p=1),
        #     'jpeg_compression': A.JpegCompression(kwargs['jpeg'], kwargs['jpeg'], always_apply=True, p=1), # 5,10,20
        #     'gaussian_noise': A.GaussNoise(mean=0, var_limit=kwargs['var'], p=1), # 5000, 50000, 500000
        #     'motion_blur': A.MotionBlur(blur_limit=(kwargs['blur'],kwargs['blur']), p=1), # 5,9,15
        #     'cartoon': iaa.Cartoon(),
        #     'mean_shift_blur': iaa.MeanShiftBlur(spatial_radius=5, color_radius=kwargs['color']), # c=1,10,50
        #     'light': iaa.MultiplyBrightness(kwargs['light']), # 0.3,0.7,1.3,1.7
        #     'sobel': Sobel(),
        #     'canny': Canny(),
        # }
        self.get_degradation(**kwargs)

        del seg_json

    def get_crop(self, img, shape, bbox, seg, scale, input=False): # , scale=(1.2, 3), ratio=(3.0/4.0, 4.0/3.0)
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

    def cos_similarity(self, vec1, vec2, cat, num):
        if len(vec2.shape) == 1:
            vec2 = vec2.unsqueeze(0)

        cos_sim = F.cosine_similarity(vec1, vec2, dim=1)

        if len(cos_sim) > num:
            max_id = torch.topk(cos_sim, k=num+1).indices
            max_id = max_id[1:]
            min_id = torch.topk(cos_sim, k=num+1, largest=False).indices
            min_id = min_id[1:]
        else:
            max_id = torch.multinomial(cos_sim, num_samples=num, replacement=True)
            min_id = torch.multinomial(cos_sim, num_samples=num, replacement=True)

        max_id += self.per_cat_pool[()][str(cat)][0]
        min_id += self.per_cat_pool[()][str(cat)][0]

        return max_id, min_id

    def mix_cos_similarity(self, vec1, vec2, cat, num, select):
        if len(vec2.shape) == 1:
            vec2 = vec2.unsqueeze(0)

        cos_sim = F.cosine_similarity(vec1, vec2, dim=1)

        # import pdb; pdb.set_trace()

        select = select.split('-')[1:]

        if len(cos_sim) > num:
            max_id = torch.topk(cos_sim, k=int(select[0])+1).indices
            max_id = max_id[1:]
            min_id = torch.topk(cos_sim, k=int(select[1])+1, largest=False).indices
            min_id = min_id[1:]
        else:
            max_id = torch.multinomial(cos_sim, num_samples=int(select[0]), replacement=True)
            min_id = torch.multinomial(cos_sim, num_samples=int(select[1]), replacement=True)

        max_id += self.per_cat_pool[()][str(cat)][0]
        min_id += self.per_cat_pool[()][str(cat)][0]

        return torch.cat([max_id, min_id])

    def get_refer_record(self, cat_id, img_id):
        try:
            ids = random.sample(self.per_cat_pool[()][str(cat_id)], k=self.num_samples+1)
        except:
            ids = random.choices(self.per_cat_pool[()][str(cat_id)], k=self.num_samples+1)

        sample_records = [self.ds[()][str(id)] for id in ids]
        img_ids = [re['image_id'] for re in sample_records]

        if img_id in img_ids:
            idx = img_ids.index(img_id)
            sample_records.pop(idx)

        return sample_records[:self.num_samples]

    def mask_transform(self, mask):
        if isinstance(mask, np.ndarray):
            if len(mask.shape) == 3:
                mask = Image.fromarray(np.squeeze(mask, axis=-1))
            else:
                mask = Image.fromarray(mask)
        return torch.LongTensor(np.array(self.output_transform(mask)).astype('int8'))

    def get_degradation(self, **kwargs):
        if self.select_type == 'gaussian':
            self.degradation = T.GaussianBlur(kernel_size=(int(kwargs['kernel']), int(kwargs['kernel'])), sigma=50)
        elif self.select_type == 'gray':
            self.degradation = T.Grayscale(3)
        elif self.select_type == 'color_jitter':
            self.degradation = T.ColorJitter(brightness=(kwargs['bright'],kwargs['bright']), contrast=(kwargs['contrast'],kwargs['contrast']), saturation=(kwargs['saturation'],kwargs['saturation']), hue=kwargs['hue'])
        elif self.select_type == 'sharp':
            self.degradation = T.RandomAdjustSharpness(kwargs['sharp'], p=1)
        elif self.select_type == 'horizontal_flip':
            self.degradation = T.RandomHorizontalFlip(p=1)
        elif self.select_type == 'vertical_flip':
            self.degradation = T.RandomVerticalFlip(p=1)
        elif self.select_type == 'posterize':
            self.degradation = T.RandomPosterize(bits=int(kwargs['bit']), p=1)
        elif self.select_type == 'solarize':
            self.degradation = T.RandomSolarize(threshold=kwargs['threshold'], p=1)
        elif self.select_type == 'equalize':
            self.degradation = T.RandomEqualize(p=1)
        elif self.select_type == 'jpeg_compression':
            self.degradation = A.JpegCompression(kwargs['jpeg'], kwargs['jpeg'], always_apply=True, p=1)
        elif self.select_type == 'gaussian_noise':
            self.degradation = A.GaussNoise(mean=0, var_limit=kwargs['var'], p=1)
        elif self.select_type == 'motion_blur':
            self.degradation = A.MotionBlur(blur_limit=(int(kwargs['blur']),int(kwargs['blur'])), p=1)
        elif self.select_type == 'cartoon':
            self.degradation = iaa.Cartoon()
        elif self.select_type == 'mean_shift_blur':
            self.degradation = iaa.MeanShiftBlur(spatial_radius=5, color_radius=kwargs['color'])
        elif self.select_type == 'light':
            self.degradation = iaa.MultiplyBrightness(kwargs['light'])
        elif self.select_type == 'sobel':
            self.degradation = Sobel()
        elif self.select_type == 'canny':
            self.degradation = Canny()
        elif self.select_type in ["bbox", "binary", "dilate", "erode"]:
            self.degradation = None
        else:
            raise TypeError
            
    def __getitem__(self, index):
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

            if self.cat in ['target', 'all']:
                if self.select_type in ['gaussian', 'gray', 'color_jitter', 'sobel', 'canny']:
                    # import pdb; pdb.set_trace()
                    refer_crop_img = refer_crop_img/255
                    if self.select_type == 'sobel':
                        refer_crop_img = T.Grayscale()(refer_crop_img)
                    if self.select_type in ['sobel', 'canny']:
                        refer_crop_img = self.degradation(refer_crop_img.unsqueeze(0))
                        refer_crop_img = refer_crop_img.squeeze(0).repeat(3,1,1)
                        refer_crop_img /= torch.max(refer_crop_img)   
                    else:
                        refer_crop_img = self.degradation(refer_crop_img)
                    
                    refer_crop_img = (refer_crop_img/torch.max(refer_crop_img)*255).to(torch.int16)

        refer = {}
        refer['input'] = refer_crop_img
        refer['output'] = refer_crop_mask
        refer['img_id'] = refer_img_id

        samples = {}
        samples['input'] = []
        samples['output'] = []
        samples['img_id'] = []

        # import pdb; pdb.set_trace()
        sample_records = self.get_refer_record(refer_cat_id, refer_img_id)
        for i in range(self.num_samples):
            sample_record = sample_records[i]
            sample_img_id = sample_record['image_id']
            sample_img_path = sample_record["file_name"]
            sample_img_shape = [sample_record["width"], sample_record["height"]]
            sample_segs = [ann["segmentation"] for ann in sample_record["annotations"]]
            sample_bboxs = [ann["bbox"] for ann in sample_record["annotations"]]

            sample_img = Image.open(sample_img_path).convert("RGB")
            sample_crop_img, sample_crop_mask = self.get_crop(np.array(sample_img), sample_img_shape, sample_bboxs, sample_segs, scale=1)

            if self.cat in ['refer', 'all']:
                if self.select_type == 'random':
                    ids = random.sample(self.per_cat_pool[()][str(refer_cat_id)], k=1)
                    random_record = [self.ds[()][str(id)] for id in ids]
                    sample_img_path = random_record[0]["file_name"]
                
                    sample_img = Image.open(sample_img_path).convert("RGB")
                    sample_crop_img, _ = self.get_crop(np.array(sample_img), sample_img_shape, sample_bboxs, sample_segs, scale=1)

                if self.select_type in ['jpeg_compression', 'gaussian_noise', 'motion_blur']:
                    sample_crop_img = self.degradation(image=np.array(sample_crop_img))['image']
                    sample_crop_img = Image.fromarray(sample_crop_img)
                
                if self.select_type in ['cartoon', 'mean_shift_blur', 'light']:
                    sample_crop_img = self.degradation(image=np.array(sample_crop_img))
                    sample_crop_img = Image.fromarray(sample_crop_img)
                
                if self.select_type in ['posterize', 'solarize', 'equalize']:
                    sample_crop_img = self.degradation(sample_crop_img)

                if self.transforms is not None:
                    sample_crop_img, sample_crop_mask = self.transforms(sample_crop_img, sample_crop_mask)

                    if self.select_type == 'binary':
                        sample_crop_img = sample_crop_mask.clone().detach()
                        sample_crop_img = sample_crop_img

                    if self.select_type in ['gaussian', 'gray', 'color_jitter', 'horizontal_flip', 'vertical_flip', 'sharp', 'sobel', 'canny']:
                        if self.select_type == 'sobel':
                            sample_crop_img = T.Grayscale()(sample_crop_img)
                        if self.select_type in ['sobel', 'canny']:
                            sample_crop_img = self.degradation(sample_crop_img.unsqueeze(0))
                            sample_crop_img = sample_crop_img.squeeze(0).repeat(3,1,1)
                            sample_crop_img /= torch.max(sample_crop_img)   
                        elif self.select_type in ['horizontal_flip', 'vertical_flip']:
                            sample_crop_img = self.degradation(sample_crop_img)
                            sample_crop_mask = self.degradation(sample_crop_mask)
                        else:
                            sample_crop_img = self.degradation(sample_crop_img)

                    elif self.select_type in ['dilate']:
                        sample_crop_mask = dilate(sample_crop_mask, ksize=10)
                    elif self.select_type in ['erode']:
                        sample_crop_mask = erode(sample_crop_mask, ksize=10)  

                    if self.select_type == 'scale':
                        scale = random.uniform(self.scale[0], self.scale[1])

                    if self.select_type == 'colormap':
                        scale = random.uniform(0, 1)
                        sample_crop_mask = sample_crop_mask * scale

                    if self.select_type == 'bbox':
                        idx = torch.where(sample_crop_mask > 0)
                        y_min, y_max = torch.min(idx[1]).item(), torch.max(idx[1]).item()
                        x_min, x_max = torch.min(idx[2]).item(), torch.max(idx[2]).item()

                        sample_crop_mask = torch.zeros_like(sample_crop_mask)
                        sample_crop_mask[:, y_min:y_max, x_min:x_max] = 1
            else:
                sample_img = Image.open(sample_img_path).convert("RGB")
                sample_crop_img, sample_crop_mask = self.get_crop(np.array(sample_img), sample_img_shape, sample_bboxs, sample_segs, scale=1)

                if self.transforms is not None:
                    sample_crop_img, sample_crop_mask = self.transforms(sample_crop_img, sample_crop_mask)

            samples['input'].append(sample_crop_img)
            samples['output'].append(sample_crop_mask)
            samples['img_id'].append(sample_img_id)

        return dict(samples=samples,
                    refer=refer,
                    name='COCO_INC_S',
                    num_samples=self.num_samples
                    )

    def __len__(self):
        return len(self.ds[()])

class SAMSegScaleDataset(VisionDataset):
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
        scale=1.0, 
    ):
        super().__init__(root, transforms, transform, target_transform)
        with open(annFile,'r') as fr: 
            seg_json = json.load(fr) 
        self.ds, self.per_cat_pool = np.array(seg_json['dataset_dicts']), np.array(seg_json['per_cat_pool'])
        self.coco_gtav_id_map = {10: 6, 1: 11, 3: 13, 8: 14, 6: 15, 7: 16, 4: 17, 2: 18}
        self.size = size
        self.num_samples = num_samples
        self.cat = cat
        self.scale = float(scale)
        self.ratio = (3.0/4.0, 4.0/3.0)
        # self.crop = RandomResizedCrop(self.size, scale=self.scale, ratio=self.ratio)
        self.sam_transform = ResizeLongestSide(self.img_size)
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
    
        # if img_min_len >= bbox_max_len:
        #     crop_len = crop_w = crop_h = img_min_len
        #     crop_x = random.randint(int(max(0, x+w-crop_len)), int(min(x, img_w-crop_len)))
        #     crop_y = random.randint(int(max(0, y+h-crop_len)), int(min(y, img_h-crop_len)))
        # else:
        #     crop_x = random.randint(0, int(x))
        #     crop_y = random.randint(0, int(y))
        #     crop_x2 = random.randint(int(x + w), img_w)
        #     crop_y2 = random.randint(int(y + h), img_h)
        #     crop_w = crop_x2 - crop_x
        #     crop_h = crop_y2 - crop_y

        if img_min_len >= bbox_max_len:
            if bbox_max_len<=self.size: # 直接裁剪
                crop_len = crop_w = crop_h = self.size
                if img_min_len < crop_len:
                    crop_len = img_min_len

                if int(max(0, x+w-crop_len)) == int(min(x, img_w-crop_len)):
                    crop_x = int(min(x, img_w-crop_len))
                else:
                    crop_x = random.randint(int(max(0, x+w-crop_len)), int(min(x, img_w-crop_len)))

                if int(max(0, y+h-crop_len)) == int(min(y, img_h-crop_len)):
                    crop_y = int(min(y, img_h-crop_len))
                else:
                    crop_y = random.randint(int(max(0, y+h-crop_len)), int(min(y, img_h-crop_len)))
            else: # 缩小后裁剪
                crop_len = crop_w = crop_h = random.randint(int(bbox_max_len), int(img_min_len))
                if int(max(0, x+w-crop_len)) == int(min(x, img_w-crop_len)):
                    crop_x = int(min(x, img_w-crop_len))
                else:
                    crop_x = random.randint(int(max(0, x+w-crop_len)), int(min(x, img_w-crop_len)))

                if int(max(0, y+h-crop_len)) == int(min(y, img_h-crop_len)):
                    crop_y = int(min(y, img_h-crop_len))
                else:
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

    def get_scale_crop(self, img, mask):
        img_h, img_w  = img.shape[1:]

        sample_mask = mask > 0.5
        idx = torch.nonzero(sample_mask[0], as_tuple=False)
        if len(idx[:, 0]) == 0:
            idx = torch.nonzero(mask[0], as_tuple=False)


        x1, y1, x2, y2 = torch.min(idx[:, 0]), torch.min(idx[:, 1]), torch.max(idx[:, 0]), torch.max(idx[:, 1])
        x_c, y_c =  torch.floor((x1 + x2)/2), torch.floor((y1 + y2)/2)
        w = torch.max(self.scale*(x2 - x1), self.scale*(y2 - y1))

        if w <=0:
            w = torch.tensor(1)

        x1_new, y1_new, x2_new, y2_new = max(0, int(x_c - w/2)), max(0, int(y_c - w/2)), min(img_w, int(x_c + w/2)), min(img_h, int(y_c + w/2))

        crop_img = img[:, x1_new:x2_new, y1_new:y2_new]
        crop_mask = sample_mask[:, x1_new:x2_new, y1_new:y2_new]

        return self.output_transform(crop_img), self.output_transform(crop_mask)

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

        sample_records = [self.ds[()][str(id)] for id in ids]
        img_ids = [re['image_id'] for re in sample_records]

        if img_id in img_ids:
            idx = img_ids.index(img_id)
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

        refer = {}
        refer['input'] = refer_crop_img
        refer['output'] = refer_crop_mask
        refer['img_id'] = refer_img_id

        samples = {}
        samples['input'] = []
        samples['output'] = []
        samples['img_id'] = []

        sample_records = random.choices(self.per_cat_pool[()][str(refer_cat_id)], k=self.num_samples)
        for i in range(self.num_samples):
            sample_record = sample_records[i]
            sample_img = Image.open(sample_record['img']).convert("RGB")
            sample_mask = Image.open(sample_record['mask']).convert("RGB")

            # label = Image.open(sample_record['mask']).convert("RGB")
            # sample_mask = np.zeros_like(sample_img, dtype=np.uint8)
            # sample_mask[np.where(np.array(label) == self.coco_gtav_id_map[refer_cat_id])] = 255
            # sample_mask = Image.fromarray(sample_mask)

            if self.transforms is not None:
                # sample_crop_img = self.sam_transform.apply_image(np.array(sample_crop_img))
                # sample_crop_img = torch.from_numpy(sample_crop_img).permute(2, 0, 1).contiguous()
                # sample_crop_mask = self.target_transform(sample_crop_mask)
                sample_img, sample_mask = self.transforms(sample_img, sample_mask)
                sample_img, sample_mask = self.get_scale_crop(sample_img, sample_mask)

            samples['input'].append(sample_img)
            samples['output'].append(sample_mask)
            samples['img_id'].append(sample_record['img'])

        return dict(samples=samples,
                    refer=refer,
                    name='COCO_INC_S',
                    num_samples=self.num_samples
                    )

    def __len__(self):
        return len(self.ds[()])

def generate_clip_region_embedding():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load("ckpts/clip/vit-L/default/ViT-L-14.pt", device=device)
    val_file = "/home/dmsheng/datasets/coco/annotations/val_seg_3w.json"

    with open(val_file,'r') as fr: 
        seg_json = json.load(fr)   
    ds, per_cat_pool = seg_json['dataset_dicts'], seg_json['per_cat_pool']

    clip_region_embeddings = dict()
    
    for k, record in tqdm(ds.items()):
        cat_id = record['category_id']
        img_path = record["file_name"]
        seg = [ann["segmentation"] for ann in record["annotations"]]

        seg = mask_util.merge(seg)
        mask = mask_util.decode([seg])
        img = np.array(Image.open(img_path).convert("RGB"))

        masked_img = mask*img

        image = preprocess(Image.fromarray(masked_img)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image)
        
        clip_region_embeddings[int(k)] = image_features
    
    import pickle
    pickle.dump(clip_region_embeddings, file=open('analysis/cos_sim/clip_region_embeddings.pkl', 'wb+'))

@torch.no_grad()
def generate_caption_analysis(model, model_type, data, data_type, device, num):
    samples, refer = data['samples'], data['refer']
    sample_imgs, sample_txts, refer_img, gt_txt = samples['input'], samples['output'], refer['input'], refer['output']
    sample_ids, gt_id = samples['img_id'], refer['img_id']

    refer_img = refer_img.to(device, non_blocking=True)
    refer_img_tokens = model.tokenizer.img_tokenizer.get_codebook_indices(refer_img).flatten(1)
    B, N = refer_img_tokens.shape
    # print(refer_img_tokens.shape)
    refer_img_emb = model.tokenizer.img_tokenizer.quantize.get_codebook_entry(refer_img_tokens, (B,int(sqrt(N)),int(sqrt(N)),256))
    q_refer_img = model.tokenizer.img_tokenizer.decode(refer_img_emb).cpu()
    q_refer_img = torch.clip(q_refer_img.detach().cpu() *255, 0, 255)

    q_img_pairs = []
    img_pairs = []

    for i in range(len(sample_imgs)):
        sample_img = sample_imgs[i]
        img_pairs.append(torch.clip(sample_img *255, 0, 255))
        
        sample_img = sample_img.to(device, non_blocking=True)
        sample_img_tokens = model.tokenizer.img_tokenizer.get_codebook_indices(sample_img).flatten(1)
        sample_img_emb = model.tokenizer.img_tokenizer.quantize.get_codebook_entry(sample_img_tokens, (B,int(sqrt(N)),int(sqrt(N)),256))
        q_sample_img = model.tokenizer.img_tokenizer.decode(sample_img_emb).cpu()
        q_sample_img = torch.clip(q_sample_img.detach().cpu() *255, 0, 255)

        q_img_pairs.append(q_sample_img)

    q_img_pairs.append(q_refer_img)
    img_pairs.append(torch.clip(refer_img.detach().cpu() *255, 0, 255))
    sample_ids.append(gt_id)

    q_imgs = torch.cat(q_img_pairs, dim=-1)
    imgs = torch.cat(img_pairs, dim=-1)
    ids = torch.stack(sample_ids)

    if data_type in ["vg", "ref"]:
        txts = sample_txts
        txts.append(gt_txt)

        if model_type == 'gpt_moe':
            special_token_ids = {'pad_token_id': model.tokenizer.txt_tokenizer.sep_token_id + model.img_token_num,
                                'eos_token_id': model.tokenizer.txt_tokenizer.eos_token_id + model.img_token_num}
            bad_words = torch.arange(1024)
            captions = model.inference(data, data_type, special_token_ids=special_token_ids, bad_words=bad_words) # , sample=True
            # captions = model.generate_caption(data, data_type, max_length=65)
        else:
            captions = model.generate(data, data_type, max_length=20)
        return ids, imgs, txts, captions
    else:

        bbox_txts = []
        caption_txts = []
        txts = []

        # import pdb; pdb.set_trace()
        for i in range(len(sample_imgs)):
            sample_txt = sample_txts[i]
            if config.model.use_map:
                sample_bbox_txt = [model.decode_caption(t.split(' Caption: ')[0]) for t in sample_txt]
                sample_caption_txt = [pre_caption(t.split('Caption: ')[-1]) for t in sample_txt]
                txt = [pre_caption(model.decode_caption(t)) for t in sample_txt]
            else:
                sample_bbox_txt = [list(map(int, t.split('[')[-1].split(']')[0].replace(" ", "").split(','))) for t in sample_txt]
                sample_caption_txt = [pre_caption(t.split('Caption: ')[-1]) for t in sample_txt]
                txt = sample_txt
            bbox_txts.append(sample_bbox_txt)
            caption_txts.append(sample_caption_txt)
            txts.append(txt)

        if config.model.use_map:
            gt_bbox_txt = [model.decode_caption(t.split(' Caption: ')[0]) for t in gt_txt]
            gt_caption_txt = [pre_caption(t.split('Caption: ')[-1]) for t in gt_txt]
        else:
            gt_bbox_txt = [list(map(int, t.split('[')[-1].split(']')[0].replace(" ", "").split(','))) for t in gt_txt]
            gt_caption_txt = [pre_caption(t.split('Caption: ')[-1]) for t in gt_txt]        
        bbox_txts.append(gt_bbox_txt)
        caption_txts.append(gt_caption_txt)

        captions = model.generate(data, data_type, max_length=65)

        if config.model.use_map:
            result_bbox_txts = [t.split(' caption ')[0] for t in captions]
            result_caption_txts = [t.split(' caption ')[-1].split('[EOS]')[0].split('.')[0] for t in captions]
            gt_txt = [pre_caption(model.decode_caption(t)) for t in sample_txt]
        else:
            result_bbox_txts = []
            for idx, t in enumerate(captions):
                try:
                    result_bbox_txts.append(list(map(int, t.split('[')[-1].split(']')[0].replace(" ", "").split(','))))
                except:
                    result_bbox_txts.append(gt_bbox_txt[idx])

            result_caption_txts = [pre_caption(t.split('Caption: ')[-1]) for t in captions]

        txts.append(gt_txt)

        return ids, imgs, bbox_txts, caption_txts, txts, result_bbox_txts, result_caption_txts, captions
        
def cos_similarity(vec1, vec2):
    cos_sim = F.cosine_similarity(vec1, vec2, dim=1)

    max_fea, max_id = torch.topk(cos_sim, k=1)
    min_fea, min_id = torch.topk(cos_sim, k=1, largest=False)

    return max_id, min_id

def eval_result(path):
    dir_names = os.listdir(path)

    metrics = {n:{} for n in dir_names}

    for n in tqdm(iter(dir_names)):
        vg_result = os.path.join(path, n, 'vg_result.json')
        vg_gt = os.path.join(path, n, 'vg_gt.json')
        coco_eval = coco_caption_eval(vg_gt, vg_result, vis=False)
        for metric, score in coco_eval.eval.items():
            metrics[n][metric] = score
    
    return metrics

if __name__ == '__main__':
    parser = get_args_parser()
    args, kwags = parser.parse_known_args()
    keys = [k.split('--')[-1] for k in kwags[::2]]
    values = [float(v) for v in kwags[1::2]]

    kwags = {k:values[i] for i, k in enumerate(keys)}

    # import pdb; pdb.set_trace()
    # generate_cos_sim_matrix()
    # generate_clip_region_embedding()
    # generate_ref_clip_region_embedding()

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    setup_seed(2333)

    save_path = os.path.join('/mnt/data/homes/dmsheng/icl_seg_results', args.ckpt.split('/')[-2], args.ckpt.split('/')[-1], args.val_path)
    if args.category:
        save_path = os.path.join(save_path, args.category + '_sample' + str(args.samples_num))
    if args.obj_size:
        save_path += f'-{args.obj_size}'

    print(save_path)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.config_path)
    if args.model:
        model = get_model(args, config)
    else:
        model = get_model(config.args, config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.ckpt, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    
    model.to(device)
    model.eval()

    val_file = "/home/dmsheng/datasets/coco/annotations/val_seg_3w.json"
    image_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor()])
    mask_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.Grayscale(3),
        transforms.ToTensor()])

    if args.data_type == 'clip':
        clip_features = pickle.load(file=open('analysis/cos_sim/clip_region_embeddings.pkl', 'rb'))
        clip_features = torch.cat([v.to(torch.float32).to('cpu') for k,v in clip_features.items()])
        dataset_val = SAMSegSimDataset(COCO_ROOT_VAL, val_file, 
                                        transform=image_transform, 
                                        target_transform=mask_transform,
                                        num_samples=args.samples_num, 
                                        clip_features=clip_features,
                                        select_type=args.select_type
                                    ) 
    elif args.data_type == 'cosine':
        clip_features = pickle.load(file=open('analysis/cos_sim/clip_region_embeddings.pkl', 'rb'))
        clip_features = torch.cat([v.to(torch.float32).to('cpu') for k,v in clip_features.items()])
        dataset_val = SAMSegSimDataset(COCO_ROOT_VAL, val_file, 
                                        transform=image_transform, 
                                        target_transform=mask_transform,
                                        num_samples=args.samples_num, 
                                        clip_features=clip_features,
                                        select_type=args.select_type
                                    ) 
    elif args.data_type == 'degrad':
        dataset_val = SAMSegDegradDataset(COCO_ROOT_VAL, val_file, 
                                        transform=image_transform, 
                                        target_transform=mask_transform,
                                        num_samples=args.samples_num, 
                                        select_type=args.select_type,
                                        **kwags
                                    ) 
    elif args.data_type == 'scale':
        val_file = "/home/dmsheng/datasets/coco/annotations/val_sd_coco_xpaste_aug_seg_3w.json"
        dataset_val = SAMSegScaleDataset(COCO_ROOT_VAL, val_file, 
                                        transform=image_transform, 
                                        target_transform=mask_transform,
                                        num_samples=args.samples_num, 
                                        scale=args.select_type
                                    ) 
    else:
        dataset_val = get_test_dataset(args)

    if args.data_type in ['mix_seg_vg-seg']: # , 'mix_seg_vg-vg'
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            collate_fn=MixSegVg_collate
        )
    else:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )

    seg_gt_masks = []
    seg_q_gt_masks = []
    seg_re_masks = []

    id = 0
    for i, batch in enumerate(tqdm(data_loader_val)):
        if args.data_type in ['CVF', 'test']:
            data = batch[0]
        else:
            data = batch

        gt_canvas, r_canvas, gt_mask, result_mask = generate_sam_img(model, data, device, i, args) # .unsqueeze(0)            
        gt_canvas, r_canvas = gt_canvas.detach().cpu(), r_canvas.detach().cpu()
        # import pdb; pdb.set_trace()

        # save img
        for b_id in range(gt_canvas.shape[0]):
            input_img = gt_canvas[b_id].numpy().transpose([1, 2, 0])
            input_img = input_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_path, 'gt_canvas_{}.png'.format(str(i*args.batch_size+b_id).zfill(5))), input_img)

            r_img = r_canvas[b_id].numpy().transpose([1, 2, 0])
            r_img = r_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_path, 'r_canvas_{}.png'.format(str(i*args.batch_size+b_id).zfill(5))), r_img)
        
        del gt_canvas, r_canvas, gt_mask, result_mask
        torch.cuda.empty_cache()

    if args.data_type in ['seg', 'mix_seg', 'fss', 'mix_fss']:
        get_seg_metrics(seg_gt_masks, seg_q_gt_masks, seg_re_masks, save_path)
    if args.data_type in ['caption', 'vg', 'unoverlap_vg', 'calibrate_vg', 'seg_vg', 'vg_bbox_text']:
        coco_caption_eval(os.path.join(save_path, "vg_gt.json"), os.path.join(save_path, "vg_result.json"))
        if args.data_type == 'vg_bbox_text':
            coco_caption_eval(os.path.join(save_path, "vg_bbox_gt.json"), os.path.join(save_path, "vg_bbox_result.json"))
            coco_caption_eval(os.path.join(save_path, "vg_all_gt.json"), os.path.join(save_path, "vg_all_result.json"))