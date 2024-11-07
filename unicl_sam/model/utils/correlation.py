r""" Provides functions that builds/manipulates correlation tensors """
import torch
from einops import rearrange
import torch.nn.functional as F

class Correlation:

    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids=None, reverse=False):
        eps = 1e-5

        corrs = []
        re_corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb)
            re_corr = torch.bmm(support_feat.transpose(1, 2), query_feat).view(bsz, hb, wb, ha, wa)
            corr = corr.clamp(min=0)
            re_corr = re_corr.clamp(min=0)
            corrs.append(corr)
            re_corrs.append(re_corr)

        if stack_ids is not None:
            corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
            corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
            corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

            return [corr_l4, corr_l3, corr_l2]
        elif reverse:
            return corrs, re_corrs
        else:
            return corrs

    def bilateral_correlation_refine(cls, forward_corr, reverse_corr, mask):
        refine_corr = []
        for i in range(len(forward_corr)):
            f_corr = rearrange(forward_corr[i], 'b ha wa hb wb -> b ha wa (hb wb)').mean(dim=-1)
            r_corr = rearrange(reverse_corr[i], 'b ha wa hb wb -> b ha wa (hb wb)').mean(dim=-1)

            mask_i = F.interpolate(mask, r_corr.shape[1:], mode='bilinear', align_corners=True).squeeze()
            r_corr = r_corr * (mask_i > 0)

            f_corr = f_corr * (r_corr > 0)

            refine_corr.append(f_corr)
        
        return refine_corr


