import torch
import pickle

baseline_ious = pickle.load(file=open('analysis/iou/dinov2_large_miou.pkl', 'rb'))
improve_ious = pickle.load(file=open('analysis/iou/dinov2_large_deepcut_graph_cluster10_miou.pkl', 'rb'))
# difference = torch.stack(improve_ious) - baseline_ious
difference = improve_ious - baseline_ious

import pdb; pdb.set_trace()

top_k = torch.topk(difference, k=10, dim=-1)
top_nk = torch.topk(difference, k=10, dim=-1, largest=False)

print(top_k)
print(top_nk)