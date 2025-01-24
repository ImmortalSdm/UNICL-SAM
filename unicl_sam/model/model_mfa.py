from pathlib import Path
import random
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, Dinov2Backbone
import torch_geometric.transforms as T

from .losses import *
from .modeling import *
from .utils import *

loss_map = {
    'ce':Lisa_CELoss(),
    'dice':Lisa_DiceLoss(),
    'iou':IoULoss(),
    'focal':FocalLoss('binary', 0.25),
    'tversky':TverskyLoss()
}

class ICL_VRP_SAM_DINO_VitDet_FPN(nn.Module):
    def __init__(self, config, use_fp16: bool = False):
        super(ICL_VRP_SAM_DINO_VitDet_FPN, self).__init__()
        self.use_fp16 = use_fp16
        self.config = config
        self.dinov2_transform = [ResizeLongestSide(214), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self._initialize_modules(config)
        self._initialize_criterions(config)

    def _initialize_modules(self, config):
        # qformer prompt encoder
        self.icl_prompt_encoder = PromptEncoder(config.model.qformer_config)

        # vision encoder
        self.processor = AutoImageProcessor.from_pretrained(config.model.dinov2_config.type, do_rescale=False)
        if 'stage' in config.model.dinov2_config.keys():
            print('Using specific stage features.')
            self.vis_encoder = Dinov2Backbone.from_pretrained(config.model.dinov2_config.type, out_features=list(config.model.dinov2_config.stage))
        else:
            self.vis_encoder = Dinov2Backbone.from_pretrained(config.model.dinov2_config.type)

        for param in self.vis_encoder.parameters():
            param.requires_grad = False

        encoder_dim = config.model.fpn_config.out_channels
        if config.model.with_simple_fpn:
            self.fpn = SimpleFeaturePyramid(self.vis_encoder.config.hidden_size, config.model.fpn_config.out_channels, (4.0, 2.0, 1.0, 0.5), norm=config.model.fpn_config.norm_type)
            if self.config.model.fpn_config.multi_scale_fusion == 'bfp':
                self.bfp = BFP(encoder_dim, num_levels=4, refine_level=2, refine_type='non_local')

        # SAM
        self.SAM = sam_model_registry[config.model.sam_config.type](checkpoint=config.model.sam_config.ckpt)
        for param in self.SAM.parameters():
            param.requires_grad = False

        if config.model.sam_config.train_mask_decoder:
            self.SAM.mask_decoder.train()
            for param in self.SAM.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        # self.mapping1 = nn.Conv2d(encoder_dim, encoder_dim, kernel_size=1)
        if config.model.with_mask:
            if self.config.model.fpn_config.multi_scale_fusion == 'concat':
                self.mapping2 = nn.Conv1d(encoder_dim*2+1, encoder_dim, kernel_size=1)
            elif self.config.model.fpn_config.multi_scale_fusion == 'bfp':
                self.mapping2 = nn.Conv2d(encoder_dim*2+1, encoder_dim, kernel_size=1)
        else:
            if self.config.model.fpn_config.multi_scale_fusion == 'concat':
                self.mapping2 = nn.Conv1d(encoder_dim*2, encoder_dim, kernel_size=1)
            elif self.config.model.fpn_config.multi_scale_fusion == 'bfp':
                self.mapping2 = nn.Conv2d(encoder_dim*2, encoder_dim, kernel_size=1)
        if config.model.with_mlp_reweight:
            self.weight = nn.Linear(encoder_dim, 1, bias=True) 

    def _initialize_criterions(self, config):
        self.criterion = {}
        for loss_name in config.loss.keys():
            if loss_name in loss_map.keys():
                self.criterion[loss_name] = {
                    'loss':loss_map[loss_name],
                    'weight':config.loss[loss_name]
                }
            else:
                continue
        
        if self.config.model.uncertainty_config.with_aug:
            self.criterion.update({
                'graph_kl_loss': {
                    'loss': nn.KLDivLoss(reduction="batchmean", log_target=True),
                    'weight': self.config.loss.kl
                }
            })


        if self.config.model.uncertainty_config.with_cluster_loss:
            self.criterion.update({
                'graph_clu_loss': {
                    'loss': DiscriminativeLoss(delta_dist=self.config.loss.un_clu.delta_dist,
                                            norm=self.config.loss.un_clu.norm,
                                            beta=self.config.loss.un_clu.beta,
                                            gamma=self.config.loss.un_clu.gamma), 
                    'weight': self.config.loss.un_clu.weight
                }
            })

        if 'ctr' in self.config.loss.keys():
            self.criterion.update({
                'ctr_loss': {
                    'loss': VICRegL_Loss(emb_dim=config.model.fpn_config.out_channels), 
                    'weight': self.config.loss.ctr
                }
            })

    def get_image_embedding(self, sam_cropped_images, model_type):
        if model_type == 'dino':
            inputs = self.processor(images=sam_cropped_images, return_tensors="pt").to(self.SAM.device).to(torch.float16)
            # inputs = self.dinov2_transform[1](self.dinov2_transform[0].apply_image_torch(sam_cropped_images)).to(self.SAM.device).to(torch.float16)
            outputs = self.vis_encoder(**inputs) # B, 3, 256, 256

            return outputs['feature_maps'][0]
        elif model_type == 'sam':
            inputs = self.SAM.preprocess(sam_cropped_images.to(self.SAM.device))
            outputs, _ = self.SAM.image_encoder(inputs) # B, 256, 64, 64            
            return outputs
        else:
            raise ValueError(
                f"Unacceptable model type! Please choose from 'dino' or 'sam."
            )

    def mask_feature(self, support_feats, support_masks):
        masks = []
        for idx, feature in enumerate(support_feats):
            mask = F.interpolate(support_masks.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            mask = mask > 0
            support_feats[idx] = support_feats[idx] * mask
            masks.append(mask.float())
        return support_feats, masks

    def get_mask_area(self, image_feature, mask):
        feat = image_feature.flatten(1).permute(1,0).contiguous()
        mask = mask.flatten()
        mask_feat = feat[mask>0]

        if len(mask_feat) == 0:
            mask_feat = feat

        return mask_feat
    
    def mask_avg_pooling(self, features, mask):
        bsz, ch, h, w = features.shape
        if mask.shape[-2:] != (h, w):
            mask = F.interpolate(mask, (h, w), mode='bilinear', align_corners=True)
        assert features.shape[-2:] == mask.shape[-2:]
        num = torch.count_nonzero(torch.sign(mask).int().flatten(2), dim=-1)
        num = torch.where(num == 0, 1, num)
        feat = torch.sum(features.flatten(2), dim=-1) / (num + 1e-8)
        return feat.view(bsz,ch,1,1).expand(features.shape).contiguous()

    def normalize_feat(self, feat, eps=1e-8):
        return (feat - feat.min()) / (feat.max() - feat.min() + eps)
    
    def get_support_embeddings(self, support_feats, support_masks, query_feats, query_feat):
        if self.config.model.fpn_config.multi_scale_fusion == 'concat':
            support_feat = torch.cat([v.flatten(2) for k,v in support_feats.items()], dim=2)

            support_mask_feats, masks = self.mask_feature([v for k,v in support_feats.items()], support_masks[: , 0, ...].to(torch.float16).to(self.SAM.device))
            corr = Correlation.multilayer_correlation([v for k,v in query_feats.items()], support_mask_feats) # B, 768, 18, 18, 18, 18

            if self.config.model.with_mask_pooling:
                support_mask_feat =  torch.cat([self.mask_avg_pooling(v, masks[0]).flatten(2) for v in support_mask_feats], dim=2) # B, 768, 18, 18
            
            if self.config.model.no_augment:
                support_embedding = support_mask_feat
            else:
                if self.config.model.with_mask:
                    support_embedding = self.mapping2(torch.cat([support_feat, support_mask_feat, torch.cat([m.flatten(2) for m in masks], dim=2)], dim=1)) # B, 768, 18, 18
                else:
                    support_embedding = self.mapping2(torch.cat([support_feat, support_mask_feat], dim=1)) # B, 768, 18, 18

            if self.config.model.with_mlp_reweight:
                import pdb; pdb.set_trace()
                weight = self.weight(support_embedding)
        elif self.config.model.fpn_config.multi_scale_fusion == 'bfp':
            support_feat = self.bfp([v for k,v in support_feats.items()])

            support_mask_feat, mask = self.mask_feature([support_feat], support_masks[: , 0, ...].to(torch.float16).to(self.SAM.device))
            support_mask_feat = support_mask_feat[-1]
            # corr = Correlation.multilayer_correlation([query_feats], [support_mask_feat_de[-1]]) # B, 768, 18, 18, 18, 18
            corr = Correlation.multilayer_correlation([query_feat], [support_mask_feat]) # B, 768, 18, 18, 18, 18

            if self.config.model.with_mask_pooling:
                support_mask_feat = self.mask_avg_pooling(support_mask_feat, mask[0]) # B, 768, 18, 18
            
            if self.config.model.no_augment:
                support_embedding = support_mask_feat
            else:
                if self.config.model.with_mask:
                    support_embedding = self.mapping2(torch.cat([support_feat, support_mask_feat, mask[0]], dim=1)) # B, 768, 18, 18
                else:
                    support_embedding = self.mapping2(torch.cat([support_feat, support_mask_feat], dim=1)) # B, 768, 18, 18

            if self.config.model.with_mlp_reweight:
                import pdb; pdb.set_trace()
                weight = self.weight(support_embedding)

        return support_mask_feat, support_embedding, corr

    def mask_refine(self, mask, min_area):
        mask = mask.cpu().numpy()
        shape = mask.shape
        mask, changed = remove_small_regions(mask[0][0], min_area, mode="holes")
        mask, changed = remove_small_regions(mask, min_area, mode="islands")

        return torch.as_tensor(mask).reshape(shape)

    def forward(self, data, data_type='CVF', infer=False, **generate_kwargs):
        # get query embedding
        supp, supp_label, query, query_label = data['supp_image'].cuda(), data['supp_label'].cuda(), data['image'].cuda(), data['label'].cuda() 
        supp_ctr, supp_ctr_label = data['supp_ctr_image'].cuda(), data['supp_ctr_label'].cuda()
        supp, supp_label = supp.transpose(0,1).contiguous(), supp_label.transpose(0,1).contiguous()
        supp_ctr, supp_ctr_label = supp_ctr.transpose(0,1).contiguous(), supp_ctr_label.transpose(0,1).contiguous()
        query_feats = self.get_image_embedding(torch.clip(query/255., 0, 1), model_type='dino') # B, 768, 18, 18

        if self.config.model.with_simple_fpn:
            query_feats = self.fpn(query_feats)
            if self.config.model.fpn_config.multi_scale_fusion == 'concat':
                query_feat = torch.cat([v.flatten(2) for k,v in query_feats.items()], dim=2)
            elif self.config.model.fpn_config.multi_scale_fusion == 'bfp':
                query_feat = self.bfp([v for k,v in query_feats.items()])
                
        # get support embedding and correspondence matrix
        support_embeddings = []
        support_mask_feats = []
        corrs = []

        num_samples = random.randint(1, len(supp))
        for i in range(num_samples):
            supp_feats = self.get_image_embedding(supp[i].to(torch.float16), model_type='dino') # B, 768, 18, 18

            if self.config.model.with_simple_fpn:
                supp_feats = self.fpn(supp_feats)
                support_mask_feat, support_embedding, corr = self.get_support_embeddings(supp_feats, supp_label[i], query_feats, query_feat)
                if self.config.model.fpn_config.multi_scale_fusion == 'concat':
                    support_mask_feats.append(support_mask_feat)
                    support_embeddings.append(support_embedding)
                    corrs.append(torch.cat([rearrange(c, 'b ha wa hb wb -> b ha wa (hb wb)').mean(dim=-1).unsqueeze(1).flatten(2) for c in corr], dim=2))
                elif self.config.model.fpn_config.multi_scale_fusion == 'bfp':
                    support_mask_feats.append(support_mask_feat)
                    support_embeddings.append(support_embedding.flatten(2))
                    corrs.append(rearrange(corr[0], 'b ha wa hb wb -> b ha wa (hb wb)').mean(dim=-1).unsqueeze(1))

        # compute pseudo mask and get enhanced features
        avg_support_mask_feats = torch.mean(torch.stack(support_mask_feats), dim=0)

        if self.config.model.no_augment:
            query_embeddings = query_feat
        else:
            if self.config.model.with_mask:
                pseudo_mask = self.normalize_feat(torch.mean(torch.stack(corrs), dim=0))
                pseudo_mask = (pseudo_mask>0.5).float()
                query_embeddings = self.mapping2(torch.cat([query_feat, avg_support_mask_feats, pseudo_mask], dim=1)) # B, 768, 18, 18
            else:
                query_embeddings = self.mapping2(torch.cat([query_feat, avg_support_mask_feats], dim=1)) # B, 768, 18, 18

        support_embeddings = torch.cat(support_embeddings, dim=-1) # B, seq_len, 768 

        # prompt generation
        sparse_embeddings = self.icl_prompt_encoder(
            support_encoder_hidden_states=support_embeddings.permute(0,2,1),
            query_encoder_hidden_states=query_embeddings.flatten(2).permute(0,2,1),
            output_attentions=False,
        )['last_hidden_state'] # B, 50, 256

        # SAM decode with sparse prompt
        input_image = self.SAM.preprocess(data['refer']['input'].to(self.SAM.device))
        image_embeddings, _ = self.SAM.image_encoder(input_image) # B, 256, 64, 64
        dense_embeddings = self.SAM.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).to(self.SAM.device)

        pred_masks = []
        for i in range(len(image_embeddings)):
            low_res_masks, _ = self.SAM.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.SAM.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings[i].unsqueeze(0),
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            pred_mask = self.SAM.postprocess_masks(
                low_res_masks,
                input_size=data['refer']['input'].shape[-2:],
                original_size=(512, 512),
            )

            if infer:
                for k,v in generate_kwargs.items():
                    if k == 'persam_refine' and v:
                        pred_mask = self.persam_mask_refine(low_res_masks, pred_mask, image_embeddings[i].unsqueeze(0))

                    if k == 'return_logits' and not v:
                        pred_mask = (pred_mask > 0).float()

                    if k == 'mask_refine' and v:
                        pred_mask = self.mask_refine(pred_mask, 36)   

            pred_masks.append(pred_mask[:, 0])

        loss_dict = self.forward_loss(pred_masks, query_label)

        if self.config.model.uncertainty_config.uncertainty_reweighting_type == 'gate':
            loss_dict['gate'] = self.uncertainty_gate.data

        if infer:
            output = {
                'gt': query_label,
                'pred': torch.stack(pred_masks)
            }
            return output, loss_dict

        return loss_dict

    def forward_loss(self, prediction, labels, data_type='CVF'):
        loss_dict = {}
        for name, criterion in self.criterion.items():
            if name in ['dice', 'ce']:
                mask_loss = 0
                num_masks = 0
                for batch_idx in range(len(prediction)):
                    gt_mask = labels[batch_idx]
                    pred_mask = prediction[batch_idx]

                    assert (
                        gt_mask.shape[0] == pred_mask.shape[0]
                    ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                        gt_mask.shape, pred_mask.shape
                    )
                    mask_loss += (
                        criterion['loss'](pred_mask.contiguous(), gt_mask.float(), num_masks=gt_mask.shape[0])
                        * gt_mask.shape[0]
                    )
                    num_masks += gt_mask.shape[0]

                loss_dict[name] = criterion['weight'] * mask_loss / (num_masks + 1e-8)
            else:
                loss_dict[name] = criterion['loss'](prediction.contiguous(), labels.float())*criterion['weight']
        
        return loss_dict

    @torch.no_grad()
    def generate_img(self, data, return_logits: bool = False, mask_refine: bool = False, persam_refine: bool = False, **generate_kwargs):
        # get query embedding
        supp, supp_label, query, query_label = data['supp_image'].cuda(), data['supp_label'].cuda(), data['image'].cuda(), data['label'].cuda() 
        supp, supp_label = supp.transpose(0,1).contiguous(), supp_label.transpose(0,1).contiguous()
        query_feats = self.get_image_embedding(torch.clip(query/255., 0, 1), model_type='dino') # B, 768, 18, 18

        if self.config.model.with_simple_fpn:
            query_feats = self.fpn(query_feats)
            if self.config.model.fpn_config.multi_scale_fusion == 'concat':
                query_feat = torch.cat([v.flatten(2) for k,v in query_feats.items()], dim=2)
            elif self.config.model.fpn_config.multi_scale_fusion == 'bfp':
                query_feat = self.bfp([v for k,v in query_feats.items()])
                
        # get support embedding and correspondence matrix
        support_embeddings = []
        support_mask_feats = []
        corrs = []

        for i in range(len(supp)):
            supp_feats = self.get_image_embedding(supp[i].to(torch.float16), model_type='dino') # B, 768, 18, 18

            if self.config.model.with_simple_fpn:
                support_feats = self.fpn(supp_feats)
                support_mask_feat, support_embedding, corr = self.get_support_embeddings(support_feats, supp_label[i], query_feats, query_feat)
                if self.config.model.fpn_config.multi_scale_fusion == 'concat':
                    support_mask_feats.append(support_mask_feat)
                    support_embeddings.append(support_embedding)
                    corrs.append(torch.cat([rearrange(c, 'b ha wa hb wb -> b ha wa (hb wb)').mean(dim=-1).unsqueeze(1).flatten(2) for c in corr], dim=2))
                elif self.config.model.fpn_config.multi_scale_fusion == 'bfp':
                    support_mask_feats.append(support_mask_feat)
                    support_embeddings.append(support_embedding.flatten(2))
                    corrs.append(rearrange(corr[0], 'b ha wa hb wb -> b ha wa (hb wb)').mean(dim=-1).unsqueeze(1))

        # compute pseudo mask and get enhanced features
        avg_support_mask_feats = torch.mean(torch.stack(support_mask_feats), dim=0)

        if self.config.model.no_augment:
            query_embeddings = query_feat
        else:
            if self.config.model.with_mask:
                pseudo_mask = torch.mean(torch.stack(corrs), dim=0)
                query_embeddings = self.mapping2(torch.cat([query_feat, avg_support_mask_feats, pseudo_mask], dim=1)) # B, 768, 18, 18
            else:
                query_embeddings = self.mapping2(torch.cat([query_feat, avg_support_mask_feats], dim=1)) # B, 768, 18, 18

        support_embeddings = torch.cat(support_embeddings, dim=-1) # B, seq_len, 768 
        # import pdb; pdb.set_trace()

        # prompt generation
        sparse_embeddings = self.icl_prompt_encoder(
            support_encoder_hidden_states=support_embeddings.permute(0,2,1),
            query_encoder_hidden_states=query_embeddings.flatten(2).permute(0,2,1),
            output_attentions=False,
        )['last_hidden_state'] # B, 50, 256

        del query_feats, query_feat, support_feats, avg_support_mask_feats, support_mask_feats, corrs, support_embeddings, query_embeddings
        torch.cuda.empty_cache()

        # SAM decode with sparse prompt
        input_image = self.SAM.preprocess(query)
        image_embeddings, _ = self.SAM.image_encoder(input_image) # B, 256, 64, 64
        dense_embeddings = self.SAM.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).to(self.SAM.device)

        pred_masks = []
        for i in range(len(image_embeddings)):
            low_res_masks, _ = self.SAM.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.SAM.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings[i].unsqueeze(0),
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            pred_mask = self.SAM.postprocess_masks(
                low_res_masks,
                input_size=query.shape[-2:],
                original_size=query.shape[-2:],
            )

            if persam_refine:
                pred_mask = self.persam_mask_refine(low_res_masks, pred_mask, image_embeddings[i].unsqueeze(0))

            if not return_logits:
                pred_mask = pred_mask > 0

            if mask_refine:
                pred_mask = self.mask_refine(pred_mask, 36)

            pred_masks.append(pred_mask[:, 0])

        del image_embeddings, dense_embeddings, sparse_embeddings
        torch.cuda.empty_cache()
        
        return pred_masks # pseudo_mask.view(bsz, ch, ha, wa, -1).mean(dim=-1) 

    @torch.no_grad()
    def analysis(self, data, return_logits: bool = False, **generate_kwargs):
        query_feats = self.get_image_embedding(data['refer']['input'].to(self.SAM.device)/255., model_type='dino') # B, 768, 18, 18

        if self.config.model.with_simple_fpn:
            query_feats = self.fpn(query_feats)
            if self.config.model.fpn_config.multi_scale_fusion == 'concat':
                query_feat = torch.cat([v.flatten(2) for k,v in query_feats.items()], dim=2)
            elif self.config.model.fpn_config.multi_scale_fusion == 'bfp':
                query_feat = self.bfp([v for k,v in query_feats.items()])

        # get support embedding and correspondence matrix
        support_embeddings = []
        support_mask_feats = []
        corrs = []

        num_samples = random.randint(1, len(data['samples']['input']))
        for i in range(num_samples):
            support_feats = self.get_image_embedding(data['samples']['input'][i].to(torch.float16).to(self.SAM.device), model_type='dino') # B, 768, 18, 18
            if self.config.model.with_simple_fpn:
                support_feats = self.fpn(support_feats)
                support_mask_feat, support_embedding, corr = self.get_support_embeddings(support_feats, data['samples']['output'][i], query_feats, query_feat)
                if self.config.model.fpn_config.multi_scale_fusion == 'concat':
                    support_mask_feats.append(support_mask_feat)
                    support_embeddings.append(support_embedding)
                    corrs.append(torch.cat([rearrange(c, 'b ha wa hb wb -> b ha wa (hb wb)').mean(dim=-1).unsqueeze(1).flatten(2) for c in corr], dim=2))
                elif self.config.model.fpn_config.multi_scale_fusion == 'bfp':
                    support_mask_feats.append(support_mask_feat)
                    support_embeddings.append(support_embedding.flatten(2))
                    corrs.append(rearrange(corr[0], 'b ha wa hb wb -> b ha wa (hb wb)').mean(dim=-1).unsqueeze(1))

        # compute pseudo mask and get enhanced features
        avg_support_mask_feats = torch.mean(torch.stack(support_mask_feats), dim=0)

        if self.config.model.no_augment:
            query_embeddings = query_feat
        else:
            if self.config.model.with_mask:
                pseudo_mask = torch.mean(torch.stack(corrs), dim=0)
                query_embeddings = self.mapping2(torch.cat([query_feat, avg_support_mask_feats, pseudo_mask], dim=1)) # B, 768, 18, 18
            else:
                query_embeddings = self.mapping2(torch.cat([query_feat, avg_support_mask_feats], dim=1)) # B, 768, 18, 18

        support_embeddings = torch.cat(support_embeddings, dim=-1) # B, seq_len, 768 

        # prompt generation
        outputs = self.icl_prompt_encoder(
            support_encoder_hidden_states=support_embeddings.permute(0,2,1),
            query_encoder_hidden_states=query_embeddings.flatten(2).permute(0,2,1),
            output_attentions=True,
        )

        del support_embeddings, support_mask_feats, corrs, query_embeddings
        torch.cuda.empty_cache()

        # return (outputs['cross_attentions'], outputs['attentions']) 
        return outputs

    @torch.no_grad()
    def persam_mask_refine(self, low_res_masks, masks, image_embeddings):
        # Positive location prior
        # topk_xy, topk_label = persam_point_sample(masks[0][0], topk=10)
        topk_xy, topk_label = vrpsam_point_sample(masks[0][0], n=5)

        # Cascaded Post-refinement-1
        final_mask = masks
        masks = masks>0
        idx = torch.nonzero(masks[0][0])
        y, x = idx[:, 0], idx[:, 1]
        if len(y) > 0 and len(x) > 0:
            y, x = idx[:, 0], idx[:, 1]
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()
            input_box = torch.tensor([x_min, y_min, x_max, y_max])
            sparse_embeddings, dense_embeddings = self.SAM.prompt_encoder(
                points=(topk_xy, topk_label),
                boxes=input_box[None, :].to(self.SAM.device),
                masks=None, # low_res_masks
            )

            low_res_masks, scores = self.SAM.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.SAM.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True
            )       

            masks = self.SAM.postprocess_masks(
                low_res_masks,
                input_size=(512, 512),
                original_size=(512, 512),
            )

            best_idx = torch.argmax(scores, dim=-1)

            # Cascaded Post-refinement-2
            masks = masks>0
            idx = torch.nonzero(masks[:, best_idx][0][0])
            y, x = idx[:, 0], idx[:, 1]
            if len(y) > 0 and len(x) > 0:
                x_min = x.min()
                x_max = x.max()
                y_min = y.min()
                y_max = y.max()
                input_box = torch.tensor([x_min, y_min, x_max, y_max])
                sparse_embeddings, dense_embeddings = self.SAM.prompt_encoder(
                    points=(topk_xy, topk_label),
                    boxes=input_box[None, :].to(self.SAM.device),
                    masks=None, # low_res_masks[:, best_idx, ...]
                )
                masks, scores = self.SAM.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.SAM.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True
                )     

                masks = self.SAM.postprocess_masks(
                    masks,
                    input_size=(512, 512),
                    original_size=(512, 512),
                )

                best_idx = torch.argmax(scores, dim=-1)

                final_mask = masks[:, best_idx, ...]

        return final_mask

if __name__=='__main__':
    # model = torchvision.models.resnet50(pretrained=True)
    # model = torch.nn.Sequential(*list(model.children())[:-2])
    input = torch.ones((4, 3, 1024, 1024))
    mask = torch.ones((4, 1, 1024, 1024))
    mask[..., 256:430, 12:503] = 0

    processor = AutoImageProcessor.from_pretrained('ckpts/dinov2/base', do_rescale=False)
    vis_encoder = AutoModel.from_pretrained('ckpts/dinov2/base')

    import pdb; pdb.set_trace()
    input = model(input)

    transpose1 = nn.ConvTranspose2d(2048, 512, kernel_size=32, stride=2)
    transpose2 = nn.ConvTranspose2d(512, 128, kernel_size=32, stride=2)
    transpose3 = nn.ConvTranspose2d(128, 64, kernel_size=24, stride=1)
    transpose4 = nn.ConvTranspose2d(64, 64, kernel_size=16, stride=1)
    mid = transpose1(input)
    mid = transpose2(mid)
    mid = transpose3(mid)
    mid = transpose4(mid)

    pool = nn.AvgPool2d(kernel_size=32)
    out = pool(input)
    # from torchsummary import summary
    # summary(model.cuda(),(3, 1024, 1024))
    # transpose1 = nn.ConvTranspose2d(3, 3, kernel_size=32, stride=2)
    # transpose2 = nn.ConvTranspose2d(3, 3, kernel_size=32, stride=2)
    # transpose3 = nn.ConvTranspose2d(3, 3, kernel_size=39, stride=1)
    # mid = transpose1(input)
    # mid = transpose2(mid)
    # mid = transpose3(mid)

    # pool = nn.AvgPool2d(kernel_size=64)
    # output = pool(mid)

    import pdb; pdb.set_trace()