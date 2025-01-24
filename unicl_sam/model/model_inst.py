from math import sqrt
from pathlib import Path
import random
import re
from functools import reduce
from operator import add
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
import torchvision.transforms as transforms
from torchvision.utils import save_image
from typing import Any, Optional, Tuple, Union

from transformers import AutoImageProcessor, Dinov2Backbone
import torch_geometric.transforms as T
from torch_geometric.data import Data, Batch
from fast_pytorch_kmeans import KMeans

from pathlib import Path
import random
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from transformers import AutoImageProcessor, Dinov2Backbone
from torch_geometric.data import Data, Batch
from fast_pytorch_kmeans import KMeans

from evaluate.evaluate_vos.memory import Memory, Frame
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

class ICL_VRP_SAM_DINO_VitDet_FPN_Uncertatinty_Deterministic_Contrastive_Inst(nn.Module):
    def __init__(self, config, use_fp16: bool = False):
        super(ICL_VRP_SAM_DINO_VitDet_FPN_Uncertatinty_Deterministic_Contrastive_Inst, self).__init__()
        self.use_fp16 = use_fp16
        self.config = config
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

        if config.model.with_simple_fpn:
            encoder_dim = config.model.fpn_config.out_channels
            self.fpn = SimpleFeaturePyramid(self.vis_encoder.config.hidden_size, config.model.fpn_config.out_channels, (4.0, 2.0, 1.0, 0.5), norm=config.model.fpn_config.norm_type)
            if self.config.model.fpn_config.multi_scale_fusion == 'bfp':
                self.bfp = BFP(encoder_dim, num_levels=4, refine_level=2, refine_type='non_local')
        else:
            encoder_dim = config.model.qformer_config.encoder_hidden_size
            self.mapping1 = nn.Conv2d(encoder_dim, encoder_dim, kernel_size=1)

        # Uncertainty estimator
        self.uncertainty_GCN = RGCN(config.model.uncertainty_config)
        # uncertainty reweighting
        if config.model.uncertainty_config.with_uncertainty_reweighting:
            if config.model.uncertainty_config.uncertainty_reweighting_type == 'gate':
                self.uncertainty_gate = torch.nn.Parameter(torch.tensor(1e-4))
            else:
                self.uncertainty_topk_ratio = config.model.uncertainty_config.uncertainty_topk_ratio
        # if self.config.model.uncertainty_config.with_cluster_loss:
        self.kmeans = KMeans(n_clusters=8, mode='euclidean', verbose=1) # , init_method='kmeans++', minibatch=4

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
            if config.model.with_simple_fpn:
                if self.config.model.fpn_config.multi_scale_fusion == 'bfp':
                    self.mapping2 = nn.Conv2d(encoder_dim*3+1, encoder_dim, kernel_size=1)
            else:
                self.mapping2 = nn.Conv2d(encoder_dim*3+1, encoder_dim, kernel_size=1)
        else:
            if config.model.with_simple_fpn:
                if self.config.model.fpn_config.multi_scale_fusion == 'bfp':
                    self.mapping2 = nn.Conv2d(encoder_dim*3, encoder_dim, kernel_size=1)
            else:
                self.mapping2 = nn.Conv2d(encoder_dim*3, encoder_dim, kernel_size=1)
                
        if config.model.with_mlp_reweight:
            self.weight = nn.Linear(encoder_dim, 1, bias=True) 

        if config.model.use_cross_inst_prompt:
            self.inst_type_embed = nn.Embedding(2, 256)

        if config.model.use_inst_proj:
            self.inst_proj = nn.Linear(encoder_dim, 256)

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

    def get_graph(self, image_features, masks, graph_type: str = 'part', return_batch: bool = False, with_aug: bool = False):
        W = []
        graph_data = []
        graph_aug_data = []
        for i in range(image_features.shape[0]):
            image_feat = rearrange(image_features[i], 'h w c->(h w) c').contiguous()
            mask = masks[i].flatten()
            mask_image_feat = image_feat[mask>0]
            if len(mask_image_feat) == 0 or graph_type=='whole':
                mask_image_feat = image_feat
            adj = gnn_util.create_adj(mask_image_feat, 0, self.config.model.uncertainty_config.alpha)

            # Data to pytorch_geometric format
            node_feats, edge_index, edge_weight = gnn_util.load_data(adj, mask_image_feat)
            if self.config.model.uncertainty_config.with_graph_pe:
                pos_embedding = self.PE(torch.arange(0, node_feats.shape[0], device=node_feats.device))
                node_feats = node_feats + pos_embedding

            data = Data(node_feats, edge_index, edge_weight)

            if with_aug:
                data = T.AddSelfLoops(attr='edge_attr')(data)

                n = np.random.randint(3)
                if n == 0:
                    data_aug = drop_nodes(data.detach().clone(), adj.detach().clone())
                elif n == 1:
                    data_aug = permute_edges(data.detach().clone())
                elif n == 2:
                    data_aug = mask_nodes(data.detach().clone())
                elif n == 3:
                    data_aug = sub_graph(data.detach().clone(), adj.detach().clone())
                
                graph_aug_data.append(data_aug)

            W.append(adj)
            graph_data.append(data)
        
        if return_batch:
            graph_data = Batch.from_data_list(graph_data).to(self.SAM.device)
            if with_aug:
                graph_aug_data = Batch.from_data_list(graph_aug_data).to(self.SAM.device)

        if with_aug:
            return graph_data, graph_aug_data
        else:
            return graph_data, W

    def get_aug_graph(self, un_pred, graph_data, with_uncertainty_sampling: bool = False):
        graph_data = T.AddSelfLoops(attr='edge_attr')(graph_data)

        n = np.random.randint(3)
        if n == 0:
            if with_uncertainty_sampling:
                try:
                    node_prob, _ = self.get_un_prob_weight(un_pred, graph_data)
                    graph_data_aug = drop_nodes(graph_data.detach().clone(), p=node_prob)
                except:
                    graph_data_aug = drop_nodes(graph_data.detach().clone())
            else:
                graph_data_aug = drop_nodes(graph_data.detach().clone())
        elif n == 1:
            if with_uncertainty_sampling:
                try:
                    _, edge_prob = self.get_un_prob_weight(un_pred, graph_data)
                    graph_data_aug = permute_edges(graph_data.detach().clone(), p=edge_prob)
                except:
                    graph_data_aug = permute_edges(graph_data.detach().clone())
            else:
                graph_data_aug = permute_edges(graph_data.detach().clone())
        elif n == 2:
            if with_uncertainty_sampling:
                try:
                    node_prob, _ = self.get_un_prob_weight(un_pred, graph_data)
                    graph_data_aug = mask_nodes(graph_data.detach().clone(), p=node_prob)
                except:
                    graph_data_aug = mask_nodes(graph_data.detach().clone())
            else:
                graph_data_aug = mask_nodes(graph_data.detach().clone())
        # elif n == 3:
        #     if with_uncertainty_sampling:
        #         node_prob, _ = self.get_un_prob_weight(un_pred, graph_data)
        #         try:
        #             graph_data_aug = subgraph(graph_data.detach().clone(), p=node_prob)
        #         except:
        #             graph_data_aug = subgraph(graph_data.detach().clone())
        #     else:
        #         graph_data_aug = subgraph(graph_data.detach().clone())
        
        return graph_data_aug    

    def get_cluster_per_graph(self, graph_data, un_pred, logits_processor: str = 'softmax'):
        if logits_processor == 'softmax':
            graph_data.x = un_pred.hard
        elif logits_processor == 'gumbel_softmax' or self.config.model.uncertainty_config.with_cluster_loss:
            graph_data.x = F.gumbel_softmax(un_pred.logits, tau=0.5, hard=True)
        else:
            raise TypeError
        clusters = Batch.to_data_list(graph_data)
        clusters = [clusters[i].x for i in range(graph_data.batch_size)]

        return clusters

    def get_cluster_feats(self, supp_feats, masks, clusters, 
                          logits_processor: str = 'softmax', 
                          return_cluster_feats: bool = True,
                          return_cluster_masks: bool = False,
                          **kwargs):

        bsz, ch, h, w = supp_feats.shape

        if return_cluster_feats:
            avg_pool_feats = supp_feats.detach().clone()

        if return_cluster_masks:
            cluster_masks = masks.detach().clone()

        if logits_processor == 'softmax':
            max_cluster = self.config.model.uncertainty_config.num_classes
        elif logits_processor == 'gumbel_softmax':
            max_cluster = clusters[0].shape[-1]
        else:
            raise TypeError
        
        cluster_feats = torch.zeros([bsz, max_cluster, ch], device=supp_feats.device)   

        for i in range(bsz):
            label = 0
            mask = (masks[i] > 0).repeat(ch, 1, 1)
            feat = supp_feats[i][mask].view(ch, -1).contiguous()
            if feat.shape[-1] == 0:
                label = 1
                feat = supp_feats[i].flatten(1)
            clu = clusters[i]
            if logits_processor == 'softmax':
                clu = F.one_hot(clu, num_classes=max_cluster)
            
            clu_id = clu.argmax(dim=-1)
            frequency = torch.bincount(clu_id, minlength=max_cluster).unsqueeze(-1).to(feat.device)
            cluster_feat = clu.permute(1,0).to(feat).contiguous() @ feat.permute(1,0).contiguous()
            cluster_feat = torch.div(cluster_feat, frequency + 1e-8)
            cluster_feats[i] = cluster_feat

            if return_cluster_feats:
                pool_feat = torch.stack([cluster_feat[id] for id in clu_id]) # n_clu, ch
                if label == 0:
                    avg_pool_feats[i][mask] = pool_feat.flatten()
                else:
                    avg_pool_feats[i][~mask] = pool_feat.flatten()
            
            if return_cluster_masks:
                mask = masks[i] > 0
                if label == 0:
                    cluster_masks[i][mask] = clu_id.float() + 1
                else:
                    cluster_masks[i][~mask] = clu_id.float() + 1

        outputs = (avg_pool_feats, )
        
        if return_cluster_feats:
            outputs = outputs + (cluster_feats,)
        if return_cluster_masks:
            outputs = outputs + (cluster_masks,)
        return outputs

    def cluster_mask_avg_pooling(self, features, clusters, masks, 
                                 return_cluster_masks: bool = False, 
                                 return_cluster_feats: bool = False, 
                                 with_graph_attn: bool = False):
        bsz, ch, h, w = features.shape
        if with_graph_attn:
            avg_pool_feats = features
        else:
            avg_pool_feats = features.detach().clone()
        cluster_masks = masks.detach().clone()
        
        if return_cluster_feats:
            max_cluster = torch.tensor([c.max() for c in clusters]).max()
            clsuter_feats = torch.zeros([bsz, max_cluster, ch])

        for i in range(bsz):
            if with_graph_attn:
                import pdb; pdb.set_trace()
                avg_pool_feats[i] = avg_pool_feats[i] * clusters[i]
                masks[i][torch.where(masks[i] > 0)] = clusters[i]
            else:
                if self.config.model.uncertainty_config.graph_type == 'whole':
                    clusters[i] = (clusters[i] + 1) * masks[i].flatten().bool()
                    clusters[i] = clusters[i][clusters[i]>0]
                    unique_elements, _ = torch.unique(clusters[i], return_inverse=True)
                else:
                    unique_elements, _ = torch.unique(clusters[i], return_inverse=True)

                mask = torch.stack(torch.where(masks[i].squeeze() > 0))
                if len(mask[0]) == 0:
                    mask = torch.stack(torch.where(masks[i].squeeze() == 0))
                for id in unique_elements:
                    cluster_id = torch.where(clusters[i] == id)[0]
                    m = mask.T[cluster_id]
                    cluster_masks[i][:, m.T[0], m.T[1]] = (id+1).float()
                    avg_pool_feats[i][:, m.T[0], m.T[1]] = torch.sum(features[i][:, m.T[0], m.T[1]], dim=-1, keepdim=True) / (len(m) + 1e-8)
                    if return_cluster_feats:
                        clsuter_feats[i, id] = avg_pool_feats[i][:, m.T[0], m.T[1]]

        out = {}
        out['clu_pool_feats'] = avg_pool_feats
        if return_cluster_masks:
            out.update({'clu_masks':cluster_masks})
        
        if return_cluster_feats:
            out.update({'clu_feats':clsuter_feats})

        return out

    def reparameterize(self, mu, logvar, k=1):
        sample_z = []
        for _ in range(k):
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            sample_z.append(eps.mul(std).add_(mu))
        sample_z = torch.cat(sample_z, dim=1)
        return sample_z

    def get_un_prob_weight(self, un_pred, graph_data, reverse: bool = False, normalize: str = 'min_max'):
        var = un_pred.var.detach().clone().to(torch.float32)
        # if normalize=='max':
        #     var /= (var.max() + 1e-8)
        # elif normalize=='min_max':
        #     var = (var - var.min()) / (var.max() - var.min() + 1e-8)

        node_prob = torch.softmax(var, dim=0)
        if reverse:
            node_prob = torch.softmax(var.max() - var, dim=0)
        node_prob_adj = gnn_util.create_adj(node_prob[..., None], 0, self.config.model.uncertainty_config.alpha)
        row, col = graph_data.edge_index[0], graph_data.edge_index[1]
        edge_prob_weight = node_prob_adj[row, col]

        # if normalize=='max':
        #     edge_prob_weight /= (edge_prob_weight.max() + 1e-8)
        # elif normalize=='min_max':
        #     edge_prob_weight = (edge_prob_weight - edge_prob_weight.min()) / (edge_prob_weight.max() - edge_prob_weight.min() + 1e-8)

        edge_prob = torch.softmax(edge_prob_weight, dim=0)

        return node_prob, edge_prob

    def get_uncertainty_estimation(self, feat, mask, 
                                   logits_processor: str = 'softmax', 
                                   refine: bool = False,
                                   **kwargs):
        # uncertainty
        if self.config.model.uncertainty_config.type == 'ugtr':
            residual = self.mapping1(feat)
            mean = self.mean_conv(feat)
            std = self.std_conv(feat)

            prob_support_feat = self.reparameterize(mean, std, 1)
            prob_support_feat_samples = self.reparameterize(mean, std, 50)
            prob_support_feat_samples = torch.sigmoid(prob_support_feat_samples)
            uncertainty = prob_support_feat_samples.var(dim=1, keepdim=True).detach()
            if self.training:
                uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
                uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
                uncertainty = F.conv2d(uncertainty, self.weight, padding=3, groups=1)
            uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
            residual *= (1 - uncertainty)
            if self.training:
                rand_mask = uncertainty < torch.Tensor(np.random.random(uncertainty.size())).to(uncertainty.device)
                residual *= rand_mask.to(torch.float32)

            mean3 = prob_support_feat_samples.mean(dim=1, keepdim=True)
            std3 = prob_support_feat_samples.var(dim=1, keepdim=True)
            std3 = (std3 - std3.min()) / (std3.max() - std3.min())

            return prob_support_feat, residual, mean3, std3, uncertainty
        else:
            graph_data, _ = self.get_graph(feat.permute(0, 2, 3, 1).contiguous(), mask, 
                                           graph_type=self.config.model.uncertainty_config.graph_type, 
                                           return_batch=True)

            pred = self.uncertainty_GCN(graph_data)
            loss = self.uncertainty_GCN.loss(pred, graph_data)

            if self.config.model.uncertainty_config.with_aug and (not refine):
                graph_aug_data = self.get_aug_graph(pred, graph_data, self.config.model.uncertainty_config.with_uncertainty_sampling)
                aug_pred = self.uncertainty_GCN(graph_aug_data)
                kl_loss = self.criterion['graph_kl_loss']['loss'](aug_pred.log_soft, pred.log_soft)*self.criterion['graph_kl_loss']['weight']
                loss['AUG_KL'] = kl_loss

            cluster = self.get_cluster_per_graph(graph_data, pred, logits_processor=logits_processor)

            return pred, cluster, loss

    def get_uncertainty_map(self, pred, mask, return_color_map: bool = False, normalize: str = 'min_max', stage='train'): # adj ratio 0.5
        from matplotlib import cm
        a_map = torch.zeros_like(mask)
        e_map = torch.zeros_like(mask)
        a_un = pred.softs.detach()
        a_map = torch.scatter(a_map.flatten(),0,torch.nonzero(mask.flatten()).flatten(),a_un).view(mask.shape)
        e_un = pred.var.detach()
        if normalize=='max':
            e_un /= e_un.max()
        elif normalize=='min_max':
            e_un = (e_un - e_un.min()) / (e_un.max() - e_un.min() + 1e-8)
        else:
            pass
        e_map = torch.scatter(e_map.flatten(),0,torch.nonzero(mask.flatten()).flatten(),e_un).view(mask.shape)

        if return_color_map:
            if stage == 'test':
                a_map = np.apply_along_axis(cm.viridis, -1, a_map.squeeze().detach().cpu().numpy())
                e_map = np.apply_along_axis(cm.viridis, -1, e_map.squeeze().detach().cpu().numpy())
                uncertain_map = torch.cat([torch.tensor(a_map[..., :3], device=mask.device), 
                                           torch.tensor(e_map[..., :3], device=mask.device)], dim=1)
        
                return torch.tensor(e_map[..., :3], device=mask.device)       
                return uncertain_map
            else:         
                a_map = np.apply_along_axis(cm.viridis, 0, a_map.squeeze().detach().cpu().numpy())
                e_map = np.apply_along_axis(cm.viridis, 0, e_map.squeeze().detach().cpu().numpy())
                uncertain_map = torch.cat([torch.tensor(a_map[:, :3], device=mask.device), 
                                           torch.tensor(e_map[:, :3], device=mask.device)], dim=3)
        
                return uncertain_map
        # elif reverse_color_map:
        #     a_map = np.apply_along_axis(cm.viridis, 0, (1-a_map).squeeze().detach().cpu().numpy())
        #     e_map = np.apply_along_axis(cm.viridis, 0, (1-e_map).squeeze().detach().cpu().numpy())
        #     uncertain_map = torch.cat([torch.tensor(a_map[:, :3], device=mask.device), 
        #                                torch.tensor(e_map[:, :3], device=mask.device)], dim=3)            
        else:
            return a_map, e_map

    def get_uncertainty_refinement(self, supp_feats, supp_masks, un_pred, refine_type: str = 'epistemic', feat_refine: bool = False, mask_refine: bool = False):
        def get_topk_mask(feat):
            value, idx = torch.unique(feat, return_inverse=True)
            k = int((len(value)-1) * self.uncertainty_topk_ratio)
            
            return torch.where(idx <= value[k], 1, 0).reshape(feat.shape)
        
        if feat_refine:
            a_map, e_map = self.get_uncertainty_map(un_pred, supp_masks)
            if self.config.model.uncertainty_config.uncertainty_reweighting_type == 'gate':
                if refine_type == 'epistemic':
                    reweight_factor = (1. - self.uncertainty_gate * e_map) # torch.tanh(torch.relu(self.uncertainty_gate))
                else:
                    reweight_factor = (1. - self.uncertainty_gate * a_map) 
                supp_feats = reweight_factor * supp_feats
                if mask_refine:
                    supp_masks = ((reweight_factor * supp_masks) > 0).type_as(supp_masks)
            elif self.config.model.uncertainty_config.uncertainty_reweighting_type == 'topk':
                mask = []
                if refine_type == 'epistemic':
                    for i in range(e_map.shape[0]):
                        mask.append(get_topk_mask(e_map[i]))
                else:
                    for i in range(e_map.shape[0]):
                        mask.append(get_topk_mask(a_map[i]))

                mask = torch.stack(mask)
                supp_feats = mask * supp_feats
                if mask_refine:
                    supp_masks = ((mask * supp_masks) > 0).type_as(supp_masks)
            else:
                if refine_type == 'epistemic':
                    mask = e_map < torch.Tensor(np.random.random(e_map.size())).to(e_map.device)
                else:
                    mask = a_map < torch.Tensor(np.random.random(a_map.size())).to(e_map.device)
                
                supp_feats = mask * supp_feats
                if mask_refine:
                    supp_masks = ((mask * supp_masks) > 0).type_as(supp_masks)

            support_mask_feats = self.mask_avg_pooling(supp_feats, supp_masks)
        else:
            support_mask_feats = self.mask_avg_pooling(supp_feats, supp_masks)

        return supp_feats, support_mask_feats, supp_masks
    
    def get_support_embeddings(self, supp_feats, supp_masks, query_feat, 
                               return_clusters: bool = False, 
                               return_uncertainty: bool = False, 
                               stage: str = 'train', 
                               **kwargs):
        supp_mask_feat, supp_mask = self.mask_feature([supp_feats], supp_masks[: , 0, ...].to(torch.float16).to(self.SAM.device))
        supp_mask_feat = supp_mask_feat[-1]
        supp_mask = supp_mask[-1]
        corr = Correlation.multilayer_correlation([query_feat], [supp_mask_feat]) # B, 768, 18, 18, 18, 18

        uncertainty_pred, cluster, uncertainty_loss = self.get_uncertainty_estimation(supp_mask_feat, supp_mask, **kwargs)

        if stage in ['test', 'val']:
            supp_mask_feat, supp_pool_feat, supp_mask = self.get_uncertainty_refinement(supp_mask_feat, supp_mask, uncertainty_pred, mask_refine=True)
        
        if self.config.model.with_mask_pooling:
            supp_mask_feat, supp_pool_feat, supp_mask = self.get_uncertainty_refinement(supp_mask_feat, supp_mask, uncertainty_pred, feat_refine=True, mask_refine=True) # , feat_refine=True
            # cluster = torch.cat([self.kmeans.fit_predict(self.get_mask_area(supp_mask_feat[i], supp_mask[i])) for i in range(supp_feats.shape[0])]).unsqueeze(0)
            if self.config.model.uncertainty_config.uncertainty_reweighting_type in ['topk', 'rand']:
                _, cluster, _ = self.get_uncertainty_estimation(supp_mask_feat, supp_mask, refine=True, **kwargs)
            if return_clusters:
                supp_clu_feat, cluster_feats, cluster_masks = self.get_cluster_feats(supp_mask_feat, supp_mask, cluster, return_cluster_masks=True, **kwargs)
            else:
                supp_clu_feat, cluster_feats = self.get_cluster_feats(supp_mask_feat, supp_mask, cluster, **kwargs)

        if self.config.model.uncertainty_config.with_cluster_loss:
            uncertainty_loss['UN_Clu'] = self.criterion['graph_clu_loss']['loss'](cluster_feats)*self.criterion['graph_clu_loss']['weight']

        if self.config.model.no_augment:
            supp_emb = supp_clu_feat
        else:
            if self.config.model.with_mask:
                supp_emb = self.mapping2(torch.cat([supp_feats, supp_pool_feat, supp_clu_feat, supp_mask], dim=1)) # B, 768, 18, 18
            else:
                supp_emb = self.mapping2(torch.cat([supp_feats, supp_pool_feat, supp_clu_feat], dim=1)) # B, 768, 18, 18

        outputs = ()
        outputs = outputs + (supp_pool_feat, supp_emb, corr, uncertainty_pred, uncertainty_loss,)
        if return_clusters:
            outputs = outputs + (cluster_masks,)
        if return_uncertainty:
            uncertain_map = self.get_uncertainty_map(uncertainty_pred, supp_mask, return_color_map=True, stage='test')
            outputs = outputs + (uncertain_map,)

        return outputs

    def mask_refine(self, mask, min_area):
        mask = mask.cpu().numpy()
        shape = mask.shape
        mask, changed = remove_small_regions(mask[0][0], min_area, mode="holes")
        mask, changed = remove_small_regions(mask, min_area, mode="islands")

        return torch.as_tensor(mask).reshape(shape)

    def extract_inst_feat(self, image_embed, inst_mask):
        is_list = isinstance(inst_mask, list)
        if is_list :
            inst_mask = None
        # HACK
        h, w = image_embed.shape[-2:]
        inst_labels_64 = F.interpolate(inst_mask.float(), (h, w), mode='bilinear', align_corners=False).squeeze(1)
        inst_labels_64 = (inst_labels_64 > 0.5).float()
        inst_embedding = torch.einsum('nchw,nhw->nc', image_embed, inst_labels_64) / inst_labels_64.sum((-1,-2)).clamp(min=1)[:, None]
        return inst_embedding

    def cal_sim(self, x, y, tau=1, eps=1e-4):
        if y.dim() == 2:
            x = x / x.norm(dim=1, keepdim=True).clamp(min=eps)
            y = y / y.norm(dim=1, keepdim=True).clamp(min=eps)
            sim = torch.einsum('bchw,bc->bhw', x, y)
        elif y.dim() == 3 :
            x = x / x.norm(dim=1, keepdim=True).clamp(min=eps)
            y = y / y.norm(dim=2, keepdim=True).clamp(min=eps)
            sim = torch.einsum('bchw,bxc->bxhw', x, y)
        elif y.dim() == 4 :
            sim = torch.einsum('bchw,bcx->bxhw', x, y.flatten(2))
        else :
            raise NotImplemented
        return sim

    def process_cross_inst_prompt(self, prompt):
        '''
        inst_prompt (n_inst, n_p, c)
        '''
        n_inst, n_p = prompt.shape[:2]
        prompt_cinst = prompt.flatten(0,1)[None].expand(n_inst, -1, -1)
        prompt_type_cinst = prompt_cinst.new_zeros(n_inst, n_inst, n_p, dtype=torch.long)
        prompt_type_cinst[range(n_inst), range(n_inst)] = 1
        prompt_type_cinst = prompt_type_cinst.flatten(1,2)

        return prompt_cinst, prompt_type_cinst

    def forward(self, data, data_type: str = 'CVF', infer=False, forward_vos=False, **generate_kwargs):
        if infer and forward_vos:
            return self.generate_vos_img(data, **generate_kwargs)
        # get query embedding
        if isinstance(data['label'], list):
            supp, supp_label, query, query_label = data['supp_image'].cuda(), data['supp_label'], data['image'].cuda(), data['label']
            if len(supp.shape) == 5:
                supp = (supp*255).squeeze(1)
                supp_label = [(x*255).squeeze(1) for x in supp_label]
                query_label = [(x*255) for x in query_label]
            
            offset_list = [len(x) for x in query_label]
            supp_offset_list = [len(x) for x in supp_label]
            assert offset_list == supp_offset_list

            query_label = torch.cat(query_label)[:, None].cuda()
            supp_label = torch.cat(supp_label)[:, None].cuda()
            query_label = ((query_label / 255) > 0.5).float()
            supp_label = ((supp_label / 255) > 0.5).float()

            supp_feats = self.get_image_embedding(torch.clip(supp/255., 0, 1).to(torch.float16), model_type='dino')
            if self.config.model.with_simple_fpn:
                supp_feats = self.fpn(supp_feats)
                if self.config.model.fpn_config.multi_scale_fusion == 'concat':
                    supp_feats = torch.cat([v.flatten(2) for k,v in supp_feats.items()], dim=2)
                elif self.config.model.fpn_config.multi_scale_fusion == 'bfp':
                    supp_feats = self.bfp([v for k,v in supp_feats.items()])
            else:
                supp_feats = self.mapping1(supp_feats)
            # supp, supp_label = supp.transpose(0,1).contiguous(), supp_label.transpose(0,1).contiguous()
            # supp_ctr, supp_ctr_label = supp_ctr.transpose(0,1).contiguous(), supp_ctr_label.transpose(0,1).contiguous()
        else:
            supp, supp_label, query, query_label = data['supp_image'].cuda(), data['supp_label'][:,:,0].cuda(), data['image'].cuda(), data['label'].cuda() 

            offset_list = [1] * len(query)
            supp_offset_list = [1] * len(query)
            assert offset_list == supp_offset_list

            supp_feats = []
            for i in range(len(supp)):
                supp_feat = self.get_image_embedding(supp[i].to(torch.float16), model_type='dino')
                if self.config.model.with_simple_fpn:
                    supp_feat = self.fpn(supp_feat)
                    if self.config.model.fpn_config.multi_scale_fusion == 'concat':
                        supp_feat = torch.cat([v.flatten(2) for k,v in supp_feat.items()], dim=2)
                    elif self.config.model.fpn_config.multi_scale_fusion == 'bfp':
                        supp_feat = self.bfp([v for k,v in supp_feat.items()])
                else:
                    supp_feat = self.mapping1(supp_feat)
                supp_feats.append(supp_feat)
            
            supp_feats = torch.cat(supp_feats)

        query_feats = self.get_image_embedding(torch.clip(query/255., 0, 1), model_type='dino') # B, 768, 18, 18
        input_image = self.SAM.preprocess(query)
        image_embeddings, _ = self.SAM.image_encoder(input_image) # B, 256, 64, 64

        if self.config.model.with_simple_fpn:
            query_feats = self.fpn(query_feats)
            if self.config.model.fpn_config.multi_scale_fusion == 'concat':
                query_feats = torch.cat([v.flatten(2) for k,v in query_feats.items()], dim=2)
            elif self.config.model.fpn_config.multi_scale_fusion == 'bfp':
                query_feats = self.bfp([v for k,v in query_feats.items()])
        else:
            query_feats = self.mapping1(query_feats)

        is_inst_list = data.get('is_inst', None)
        if is_inst_list != None:
            # extract instance feature
            supp_embeddings = torch.cat([x[None].expand(num_inst, -1, -1, -1) for x, num_inst in zip(supp_feats, supp_offset_list)])
            supp_label_split = supp_label.split(supp_offset_list)

            if self.config.model.use_inst_proj:
                inst_feat_duals = [self.extract_inst_feat(x[None], label) for x, label in zip(supp_feats, supp_label_split)]
                inst_feat_duals = torch.cat(inst_feat_duals) # 1, 1024
                supp_feat_prompt = self.inst_proj(inst_feat_duals)[:, None] # (bs, 1, c)
                supp_feat_prompt = supp_feat_prompt.split(supp_offset_list)

            supp_embeddings = supp_embeddings.split(supp_offset_list)
            query_embeddings = torch.cat([x[None].expand(num_inst, -1, -1, -1) for x, num_inst in zip(query_feats, offset_list)])
            query_embeddings = query_embeddings.split(offset_list)

        uncertainty_loss = {'KL': 0, 'L2': 0, 'AUG_KL': 0, 'UN_CE': 0, 'UN_Clu': 0}
        pred_masks = []
        for i, (query_feat, supp_inst_feat) in enumerate(zip(query_embeddings, supp_embeddings)):
            supp_mask_feat, supp_embedding, corr, _, u_loss = self.get_support_embeddings(supp_inst_feat, supp_label_split[i], query_feat) 
            corr = rearrange(corr[0], 'b ha wa hb wb -> b ha wa (hb wb)').contiguous().mean(dim=-1).unsqueeze(1)
            
            for k,v in u_loss.items():
                uncertainty_loss[k] += v

            # compute pseudo mask and get enhanced features
            if self.config.model.no_augment:
                query_embeddings = query_feat
                pseudo_mask = None
            else:
                if self.config.model.with_mask:
                    pseudo_mask = self.normalize_feat(corr)
                    pseudo_mask = (pseudo_mask>0.5).float()
                    query_pseudo_mask_feat = query_feat*pseudo_mask
                    _, query_cluster, _ = self.get_uncertainty_estimation(query_pseudo_mask_feat, pseudo_mask)
                    query_mask_feat = self.cluster_mask_avg_pooling(query_pseudo_mask_feat, query_cluster, pseudo_mask)['clu_pool_feats']
                    query_embeddings = self.mapping2(torch.cat([query_feat, supp_mask_feat, query_mask_feat, pseudo_mask], dim=1)) # B, 768, 18, 18
                else:
                    pseudo_mask = self.normalize_feat(corr)
                    pseudo_mask = (pseudo_mask>0.5).float()
                    query_embeddings = self.mapping2(torch.cat([query_feat, supp_mask_feat, query_feat*pseudo_mask], dim=1)) # B, 768, 18, 18

            # prompt generation
            sparse_embeddings = self.icl_prompt_encoder(
                support_encoder_hidden_states=supp_embedding.flatten(2).permute(0,2,1).contiguous(),
                query_encoder_hidden_states=query_embeddings.flatten(2).permute(0,2,1).contiguous(),
                output_attentions=False,
            )['last_hidden_state'] # B, 50, 256

            # SAM decode with sparse prompt
            aug_cross_inst = self.config.model.use_cross_inst_prompt and (infer or random.random() > 0.5)
            
            if self.config.model.use_inst_proj and is_inst_list[i]:
                inst_feat_prompt_this = supp_feat_prompt[i]

                if aug_cross_inst:
                    inst_feat_prompt_cinst, inst_feat_prompt_cinst_type = self.process_cross_inst_prompt(inst_feat_prompt_this) # ninst, ninst, 256; ninst, ninst
                    inst_feat_prompt_cinst = inst_feat_prompt_cinst + self.inst_type_embed(inst_feat_prompt_cinst_type) # ninst, ninst, 256
                    inst_feat_prompt_this = inst_feat_prompt_cinst
                    sparse_embeddings = torch.cat([sparse_embeddings, inst_feat_prompt_this], dim=1)

            # _, dense_embeddings = self.SAM.prompt_encoder(
            #     points=None,
            #     boxes=None,
            #     masks=F.interpolate(pseudo_mask.float(), (256, 256), mode='bilinear', align_corners=False)
            # ) 
            dense_embeddings = self.SAM.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).to(self.SAM.device)

            low_res_masks, _ = self.SAM.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.SAM.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            pred_mask = self.SAM.postprocess_masks(
                low_res_masks,
                input_size=query.shape[-2:],
                original_size=query.shape[-2:],
            )

            if infer:
                for k,v in generate_kwargs.items():
                    if k == 'persam_refine' and v:
                        pred_mask = self.persam_mask_refine(low_res_masks, pred_mask, image_embeddings[i].unsqueeze(0))

                    if k == 'return_logits' and not v:
                        pred_mask = (pred_mask > 0).float()

                    if k == 'mask_refine' and v:
                        pred_mask = self.mask_refine(pred_mask, 36)   

            pred_masks.append(pred_mask)

        pred_masks = torch.cat(pred_masks)
        loss_dict = self.forward_loss(pred_masks, query_label)
        num_samples = len(pred_masks)
        loss_dict.update({
            'UN_KL': uncertainty_loss['KL'] / num_samples,
            'UN_L2': uncertainty_loss['L2'] / num_samples
        })
        # loss_dict['UN_KL'] = uncertainty_loss['KL'] / num_samples
        # loss_dict['UN_L2'] = uncertainty_loss['L2'] / num_samples
        if self.config.model.uncertainty_config.with_aug:
            loss_dict.update({
                'AUG_KL': uncertainty_loss['AUG_KL'] / num_samples
            })
            # loss_dict['AUG_KL'] = uncertainty_loss['AUG_KL'] / num_samples
        if self.config.model.uncertainty_config.with_cluster_loss:
            loss_dict.update({
                'UN_Clu': uncertainty_loss['UN_Clu'] / num_samples
            })
            # loss_dict['UN_Clu'] = uncertainty_loss['UN_Clu'] / num_samples
        if self.config.model.uncertainty_config.uncertainty_reweighting_type == 'gate':
            loss_dict['gate'] = self.uncertainty_gate.data
  
        del image_embeddings, dense_embeddings, sparse_embeddings
        torch.cuda.empty_cache()

        if infer:
            output = {
                'gt': query_label,
                'pred': torch.stack(pred_masks)
            }
            return output, loss_dict

        return loss_dict

    def forward_loss(self, prediction, labels):
        loss_dict = {}
        for name, criterion in self.criterion.items():
            if name in ['dice', 'ce', 'tversky']:
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
            # else:
            #     loss_dict[name] = criterion['loss'](prediction.contiguous(), labels.float())*criterion['weight']
        
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
        else:
            query_feat = self.mapping1(query_feats)
                
        # get support embedding and correspondence matrix
        supp_embeddings = []
        supp_mask_feats = []
        corrs = []

        for i in range(len(supp)):
            supp_feats = self.get_image_embedding(supp[i].to(torch.float16), model_type='dino') # B, 768, 18, 18

            supp_mask_feat, supp_embedding, corr, _, _ = self.get_support_embeddings(supp_feats, supp_label[i], query_feat, **generate_kwargs) 

            supp_mask_feats.append(supp_mask_feat)
            supp_embeddings.append(supp_embedding.flatten(2))
            corrs.append(rearrange(corr[0], 'b ha wa hb wb -> b ha wa (hb wb)').contiguous().mean(dim=-1).unsqueeze(1))
            
        # compute pseudo mask and get enhanced features
        supp_mask_feats = torch.stack(supp_mask_feats)
        avg_supp_mask_feats = torch.mean(supp_mask_feats, dim=0)
        # if self.config.model.no_augment:
        #     query_embeddings = query_feat
        # else:
        #     if self.config.model.with_mask:
        #         pseudo_mask = torch.mean(torch.stack(corrs), dim=0)
        #         query_pseudo_mask_feat = query_feat*(pseudo_mask>0)
        #         _, query_cluster, _ = self.get_uncertainty_estimation(query_pseudo_mask_feat, (pseudo_mask>0))
        #         query_mask_feat = self.cluster_mask_avg_pooling(query_pseudo_mask_feat, query_cluster, (pseudo_mask>0).float())['clu_pool_feats']
        #         query_embeddings = self.mapping2(torch.cat([query_feat, avg_supp_mask_feats, query_mask_feat, pseudo_mask], dim=1)) # B, 768, 18, 18
        #     else:
        #         pseudo_mask = torch.mean(torch.stack(corrs), dim=0)
        #         query_embeddings = self.mapping2(torch.cat([query_feat, avg_supp_mask_feats, query_feat*pseudo_mask], dim=1)) # B, 768, 18, 18

        if self.config.model.no_augment:
            query_embeddings = query_feat
        else:
            if self.config.model.with_mask:
                pseudo_mask = self.normalize_feat(torch.mean(torch.stack(corrs), dim=0))
                pseudo_mask = (pseudo_mask>0.5).float()
                query_pseudo_mask_feat = query_feat*pseudo_mask
                _, query_cluster, _ = self.get_uncertainty_estimation(query_pseudo_mask_feat, pseudo_mask)
                query_mask_feat = self.cluster_mask_avg_pooling(query_pseudo_mask_feat, query_cluster, pseudo_mask)['clu_pool_feats']
                query_embeddings = self.mapping2(torch.cat([query_feat, avg_supp_mask_feats, query_mask_feat, pseudo_mask], dim=1)) # B, 768, 18, 18
            else:
                pseudo_mask = self.normalize_feat(torch.mean(torch.stack(corrs), dim=0))
                pseudo_mask = (pseudo_mask>0.5).float()
                query_embeddings = self.mapping2(torch.cat([query_feat, avg_supp_mask_feats, query_feat*pseudo_mask], dim=1)) # B, 768, 18, 18

        supp_embeddings = torch.cat(supp_embeddings, dim=-1) # B, seq_len, 768 
        # import pdb; pdb.set_trace()

        # prompt generation
        sparse_embeddings = self.icl_prompt_encoder(
            support_encoder_hidden_states=supp_embeddings.permute(0,2,1),
            query_encoder_hidden_states=query_embeddings.flatten(2).permute(0,2,1),
            output_attentions=False,
        )['last_hidden_state'] # B, 50, 256

        del query_feats, query_feat, supp_feats, avg_supp_mask_feats, corrs, supp_embeddings, query_embeddings
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
    def generate_vos_img(self, data, return_logits: bool = False, mask_refine: bool = False, persam_refine: bool = False, **generate_kwargs):
        try:
            from detectron2.structures import BitMasks, Instances

        except ImportError:
            print("Module vos.utils not found. Please ensure it is installed and available.")
        # get query embedding
        frames, inst_masks, inst_ids = data['images'].cuda(), data['inst_masks'].cuda(), data['inst_ids'].cuda()
        n_inst = inst_masks.shape[1]

        split_len = 1
        images_split = [frames[0, i:i+split_len] for i in range(0, len(frames[0]), split_len)]
        
        image_feats = []
        with torch.no_grad():
            for x in images_split:
                image_feat = self.get_image_embedding(x, model_type='dino')

                if self.config.model.with_simple_fpn:
                    image_feat = self.fpn(image_feat)
                    if self.config.model.fpn_config.multi_scale_fusion == 'concat':
                        image_feat = torch.cat([v.flatten(2) for k,v in image_feat.items()], dim=2)
                    elif self.config.model.fpn_config.multi_scale_fusion == 'bfp':
                        image_feat = self.bfp([v for k,v in image_feat.items()])
                else:
                    image_feat = self.mapping1(image_feat)

                image_feats.append(image_feat)
                torch.cuda.empty_cache()
        image_feats = torch.cat(image_feats)

        supp_inst_feats = [self.extract_inst_feat(image_feats[0][None].expand(n_inst,-1,-1,-1), inst_masks.permute(1,0,2,3))] # ninst, 1024
        inst_masks_pred = [inst_masks]

        def wrap_data_ref(batch):
            ref_dict = {}

            # ref
            ref_image_shape = batch['support_imgs'].shape[-2:]
            ref_dict["image"] = batch['support_imgs']

            # label
            ref_dict['height'], ref_dict['width'] = ref_image_shape
            ref_mask_num = batch['support_masks'].shape[1]
            ref_instances = Instances(ref_image_shape)
            ref_instances.gt_classes = batch['class_id']
            ref_masks = batch['support_masks']
            ref_instances.gt_masks = ref_masks
            ref_instances.gt_boxes = BitMasks(ref_masks[0]).get_bounding_boxes()
            ref_instances.ins_ids = batch['class_id'] # TODO
            ref_dict["instances"] = ref_instances
            ref_dict["support_prompts"] = batch["support_prompts"]

            return ref_dict

        def wrap_data_tar(batch):
            tar_dict = {}

            # tar
            tar_image_shape = batch['query_img'].shape[-2:]  # h, w
            tar_dict["image"] = batch['query_img']

            # # label
            tar_dict['height'], tar_dict['width'] = tar_image_shape
            tar_dict['resize_height'], tar_dict['resize_width'] = tar_image_shape

            return tar_dict

        def wrap_instances(mask, id):
            # ref
            ref_instances = Instances(mask.shape[-2:])
            ref_instances.gt_classes = torch.tensor([id], dtype=torch.int64)
            ref_instances.gt_masks = mask
            ref_instances.gt_boxes = BitMasks(mask[0]).get_bounding_boxes()
            ref_instances.ins_ids = torch.tensor([id], dtype=torch.int64)
            
            return ref_instances

        memory = {}
        for j in range(n_inst):
            assert j not in memory

            memory[j] = Memory(
                memory_len=generate_kwargs['num_frame'],
                fix_first_frame=generate_kwargs['fix_first_frame'],
                fix_last_frame=generate_kwargs['fix_last_frame'],
                memory_decay_type=generate_kwargs['memory_decay_type'],
                memory_decay_ratio=generate_kwargs['memory_decay_ratio']
            )
        # get support embedding and correspondence matrix

        for i in range(len(image_feats)-1):
            supp_feat = image_feats[i][None]
            query_feat = image_feats[i+1][None]

            mask_buff = []

            if self.config.model.use_inst_proj:
                supp_feat_prompt = self.inst_proj(supp_inst_feats[i])[:, None] # (ninst, bs, c)
            else:
                supp_feat_prompt = None

            if self.config.model.use_cross_inst_prompt:
                if self.config.model.use_inst_proj:
                    inst_feat_prompt_cinst, inst_feat_prompt_cinst_type = self.process_cross_inst_prompt(supp_feat_prompt) # ninst, ninst, 256; ninst, ninst
                    supp_feat_prompt = inst_feat_prompt_cinst + self.inst_type_embed(inst_feat_prompt_cinst_type)

            for j in range(n_inst):
                supp_mask = inst_masks_pred[i][:, j][None]
                class_id = torch.tensor([j], dtype=torch.int64, device=supp_feat.device)
                
                batch = dict(
                    support_imgs=supp_feat,
                    support_masks=supp_mask,
                    support_prompts=supp_feat_prompt[:, j],
                    class_id=class_id
                )

                ref_dict = wrap_data_ref(batch)

                if memory[j].last_frame == None:
                    memory[j].update_memory(
                        Frame(
                            obj=ref_dict,
                            frame_id=i,
                            score=1.
                        )
                    )

                supp_mask_feats = []
                supp_embeddings = []
                supps = [frame.obj for frame in memory[j].get_memory()]
                for supp in supps:
                    supp_feat, supp_mask = supp['image'], supp['instances'].gt_masks
                    supp_mask_feat, supp_embedding, corr, _, _ = self.get_support_embeddings(supp_feat, supp_mask, query_feat, **generate_kwargs) 
                    supp_mask_feats.append(supp_mask_feat)
                    supp_embeddings.append(supp_embedding.flatten(2))

                supp_mask_feats = torch.stack(supp_mask_feats)
                avg_supp_mask_feats = torch.mean(supp_mask_feats, dim=0)

                # compute pseudo mask and get enhanced features
                if self.config.model.no_augment:
                    query_embeddings = query_feat
                else:
                    if self.config.model.with_mask:
                        pseudo_mask = self.normalize_feat(rearrange(corr[0], 'b ha wa hb wb -> b ha wa (hb wb)').mean(dim=-1).unsqueeze(1))
                        pseudo_mask = (pseudo_mask>0.5).float()
                        query_pseudo_mask_feat = query_feat*pseudo_mask
                        _, query_cluster, _ = self.get_uncertainty_estimation(query_pseudo_mask_feat, pseudo_mask)
                        query_mask_feat = self.cluster_mask_avg_pooling(query_pseudo_mask_feat, query_cluster, pseudo_mask)['clu_pool_feats']
                        query_embeddings = self.mapping2(torch.cat([query_feat, avg_supp_mask_feats, query_mask_feat, pseudo_mask], dim=1))
                    else:
                        pseudo_mask = self.normalize_feat(rearrange(corr[0], 'b ha wa hb wb -> b ha wa (hb wb)').mean(dim=-1).unsqueeze(1))
                        pseudo_mask = (pseudo_mask>0.5).float()
                        query_embeddings = self.mapping2(torch.cat([query_feat, avg_supp_mask_feats, query_feat*pseudo_mask], dim=1)) # B, 768, 18, 18

                supp_embeddings = torch.cat(supp_embeddings, dim=-1) # B, seq_len, 768 

                # prompt generation
                sparse_embeddings = self.icl_prompt_encoder(
                    support_encoder_hidden_states=supp_embeddings.permute(0,2,1),
                    query_encoder_hidden_states=query_embeddings.flatten(2).permute(0,2,1),
                    output_attentions=False,
                )['last_hidden_state'] # B, 50, 256

                supp_inst_prompt = torch.mean(torch.stack([s['support_prompts'] for s in supps]), dim=0)[None]
                sparse_embeddings = torch.cat([sparse_embeddings, supp_inst_prompt], dim=1)

                del supp_embeddings, query_embeddings
                torch.cuda.empty_cache()

                # SAM decode with sparse prompt
                input_image = self.SAM.preprocess(torch.clip(images_split[i+1]*255., 0, 255))
                image_embeddings, _ = self.SAM.image_encoder(input_image) # B, 256, 64, 64
                dense_embeddings = self.SAM.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).to(self.SAM.device)

                low_res_masks, _ = self.SAM.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.SAM.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                pred_mask = self.SAM.postprocess_masks(
                    low_res_masks,
                    input_size=frames[:, i+1].shape[-2:],
                    original_size=frames[:, i+1].shape[-2:],
                )

                if persam_refine:
                    pred_mask = self.persam_mask_refine(low_res_masks, pred_mask, image_embeddings[i].unsqueeze(0))

                if not return_logits:
                    pred_mask = pred_mask > 0

                if mask_refine:
                    pred_mask = self.mask_refine(pred_mask, 36)

                mask_buff.append(pred_mask)

                batch = dict(
                    query_img=query_feat,
                    query_mask=None,
                )

                del image_embeddings, dense_embeddings, sparse_embeddings
                torch.cuda.empty_cache()
        
            mask_buff = torch.cat(mask_buff, dim=1)
            supp_inst_feats.append(self.extract_inst_feat(query_feat.expand(n_inst,-1,-1,-1), mask_buff.permute(1,0,2,3)))
            inst_masks_pred.append(mask_buff)
        return inst_masks_pred, n_inst # pseudo_mask.view(bsz, ch, ha, wa, -1).mean(dim=-1) 

    @torch.no_grad()
    def analysis(self, data, **generate_kwargs):
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
        supp_embeddings = []
        supp_mask_feats = []
        corrs = []

        for i in range(len(supp)):
            supp_feats = self.get_image_embedding(supp[i].to(torch.float16), model_type='dino') # B, 768, 18, 18

            supp_mask_feat, supp_embedding, corr, _, _ = self.get_support_embeddings(supp_feats, supp_label[i], query_feat, **generate_kwargs) 

            supp_mask_feats.append(supp_mask_feat)
            supp_embeddings.append(supp_embedding.flatten(2))
            corrs.append(rearrange(corr[0], 'b ha wa hb wb -> b ha wa (hb wb)').mean(dim=-1).unsqueeze(1))
            
        # compute pseudo mask and get enhanced features
        supp_mask_feats = torch.stack(supp_mask_feats)
        avg_supp_mask_feats = torch.mean(supp_mask_feats, dim=0)
        if self.config.model.no_augment:
            query_embeddings = query_feat
        else:
            if self.config.model.with_mask:
                pseudo_mask = self.normalize_feat(torch.mean(torch.stack(corrs), dim=0))
                pseudo_mask = (pseudo_mask>0.5).float()
                query_pseudo_mask_feat = query_feat*pseudo_mask
                _, query_cluster, _ = self.get_uncertainty_estimation(query_pseudo_mask_feat, pseudo_mask)
                query_mask_feat = self.cluster_mask_avg_pooling(query_pseudo_mask_feat, query_cluster, pseudo_mask)['clu_pool_feats']
                query_embeddings = self.mapping2(torch.cat([query_feat, avg_supp_mask_feats, query_mask_feat, pseudo_mask], dim=1)) # B, 768, 18, 18
            else:
                pseudo_mask = self.normalize_feat(torch.mean(torch.stack(corrs), dim=0))
                pseudo_mask = (pseudo_mask>0.5).float()
                query_embeddings = self.mapping2(torch.cat([query_feat, avg_supp_mask_feats, query_feat*pseudo_mask], dim=1)) # B, 768, 18, 18

        supp_embeddings = torch.cat(supp_embeddings, dim=-1) # B, seq_len, 768 

        # prompt generation
        outputs = self.icl_prompt_encoder(
            support_encoder_hidden_states=supp_embeddings.permute(0,2,1),
            query_encoder_hidden_states=query_embeddings.flatten(2).permute(0,2,1),
            output_attentions=True,
        )

        del supp_embeddings, supp_mask_feats, corrs, query_embeddings
        torch.cuda.empty_cache()

        return (outputs['cross_attentions'], outputs['attentions']) 

    @torch.no_grad()
    def cluster_visulization(self, data, **generate_kwargs):
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
        supp_mask_feats = []
        supp_cluster_masks = []

        for i in range(len(supp)):
            supp_feats = self.get_image_embedding(supp[i].to(torch.float16), model_type='dino') # B, 768, 18, 18

            supp_mask_feat, supp_cluster_mask = self.get_support_embeddings(supp_feats, supp_label[i], query_feat, **generate_kwargs) 

            supp_mask_feats.append(supp_mask_feat)
            supp_cluster_masks.append(supp_cluster_mask)

        return torch.stack(supp_mask_feats).permute(1,0,2,3,4), torch.stack(supp_cluster_masks).permute(1,0,2,3,4)

    @torch.no_grad()
    def uncertainty_analysis(self, data, **generate_kwargs):
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
        supp_mask_feats = []
        supp_uncertainty_maps = []
        corrs = []

        for i in range(len(supp)):
            supp_feats = self.get_image_embedding(supp[i].to(torch.float16), model_type='dino') # B, 768, 18, 18

            supp_mask_feat, _, corr, u_map = self.get_support_embeddings(supp_feats, supp_label[i], query_feat, **generate_kwargs) 

            supp_mask_feats.append(supp_mask_feat)
            supp_uncertainty_maps.append(u_map)
            corrs.append(rearrange(corr[0], 'b ha wa hb wb -> b ha wa (hb wb)').mean(dim=-1).unsqueeze(1))

        return torch.stack(supp_mask_feats).permute(1,0,2,3,4), torch.stack(supp_uncertainty_maps).permute(1,0,2,3,4)

    @torch.no_grad()
    def visualize_intermediate(self, data, **generate_kwargs):
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
        supp_cluster_masks = []
        supp_uncertainty_maps = []
        corrs = []

        for i in range(len(supp)):
            supp_feats = self.get_image_embedding(supp[i].to(torch.float16), model_type='dino') # B, 768, 18, 18

            _, _, corr, _, _, supp_cluster_mask, u_map = self.get_support_embeddings(supp_feats, supp_label[i], query_feat, **generate_kwargs) 
            corrs.append(rearrange(corr[0], 'b ha wa hb wb -> b ha wa (hb wb)').mean(dim=-1).unsqueeze(1))
            supp_cluster_masks.append(supp_cluster_mask)
            supp_uncertainty_maps.append(u_map)

        # compute pseudo mask and get enhanced features
        temp_mask = self.normalize_feat(torch.mean(torch.stack(corrs), dim=0))
        pseudo_mask = (temp_mask>0.65).float()
        query_pseudo_mask_feat = query_feat*pseudo_mask
        query_pred, _, _ = self.get_uncertainty_estimation(query_pseudo_mask_feat, pseudo_mask)
        query_cluster = self.kmeans.fit_predict(self.get_mask_area(query_pseudo_mask_feat[0], pseudo_mask)).unsqueeze(0)
        _, _, query_cluster_masks = self.get_cluster_feats(query_pseudo_mask_feat, pseudo_mask, query_cluster, return_cluster_masks=True, **generate_kwargs)
        query_uncertain_map = self.get_uncertainty_map(query_pred, pseudo_mask, return_color_map=True, stage='test').permute(2,0,1).unsqueeze(0)

        return torch.stack(supp_cluster_masks), torch.stack(supp_uncertainty_maps).permute(0,3,1,2), query_cluster_masks, query_uncertain_map, temp_mask

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
    config = OmegaConf.load('configs/unicl_sam/vrp_sam_dinov2_large_vitdet_fpn_uncertainty.yaml')
    model = ICL_VRP_SAM_DINO_VitDet_FPN_Uncertatinty_Deterministic_Contrastive_Inst(config)

    model.cuda()
    from unicl_sam.data import *

    from tqdm import tqdm
    # from torchsummary import summary
    image_transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor()])
    mask_transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.Grayscale(),
        transforms.ToTensor()])

    aug_inst = get_inst_aug(518)
    aug_inst = DualAug([aug_inst])

    dataset_val = CustomConcatDataset([
            SAMImgSegDataset(transform=image_transform, target_transform=mask_transform, size=518, is_train=False, dataset_name='coco_train'), 
            SAMSegADEDataset(ADE_ROOT, transform=image_transform, target_transform=mask_transform, size=518, num_samples=1, is_train=False),
            SAMSegLVISDataset(LVIS_ROOT, transform=image_transform, target_transform=mask_transform, size=518, num_samples=1, is_train=False),
            InstCOCO(base_image_dir='/home/qchugroup/sdmcvpr2025/datasets/coco', transform=aug_inst, dataset_name='coco', is_train=False),
            InstCOCO(base_image_dir='/home/qchugroup/sdmcvpr2025/datasets/coco', transform=aug_inst, dataset_name='lvis', is_train=False)
            ],
            None, samples_per_epoch=None)

    sampler = torch.utils.data.RandomSampler(dataset_val)
    sampler_train = torch.utils.data.BatchSampler(sampler, 4, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset_val, batch_sampler=sampler_train,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=True
    )

    transform = transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.NEAREST)
    for num, data in tqdm(enumerate(data_loader)):
        # import pdb; pdb.set_trace()
        print(data['is_inst']) # 'image', 'shape', 'is_inst', 'supp_image', 'supp_shape', 'label', 'supp_label'
        loss = model(data)
        # support_mask_feat, support_uncertainty_maps = model.uncertainty_analysis(data)
        # support_img, support_mask = torch.clip(transform(data['samples']['input'][0]).detach().cpu(), 0, 255), torch.clip(transform(data['samples']['output'][0]).detach().cpu() *255, 0, 255)

        # support_uncertainty_maps = F.interpolate(support_uncertainty_maps.squeeze().detach().cpu(), (512, 1024), mode='nearest')
        # # import pdb; pdb.set_trace()
        # canvas = torch.cat([torch.cat([support_img, support_mask/255], dim=-1), support_uncertainty_maps], dim=-1)
        # save_path = 'analysis/uncertainty_map/test/test_sample50'
        # Path(save_path).mkdir(parents=True, exist_ok=True)
        # save_image(canvas, f'{save_path}/{num}.png', nrow=1)
