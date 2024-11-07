from typing import Optional
from torch_geometric.utils import subgraph, k_hop_subgraph
import torch

def drop_nodes(data, p: Optional[torch.Tensor] = None):
    node_num, _ = data.x.size()
    drop_num = int(node_num * 0.2)
    if drop_num == 0:
        drop_num = 1

    edge_index = data.edge_index
    edge_attr = data.edge_attr

    if isinstance(p, torch.Tensor):
        idx_drop = torch.multinomial(p, num_samples=drop_num, replacement=False)
    else:
        idx_drop = torch.multinomial(torch.softmax(torch.ones([node_num], device=edge_index.device), dim=0), num_samples=drop_num, replacement=False)

    idx_keep = torch.arange(node_num, device=edge_index.device)
    idx_keep = idx_keep[~torch.isin(idx_keep, idx_drop)]
    edge_index, edge_attr = subgraph(idx_keep, edge_index, edge_attr)

    data.edge_index = edge_index
    data.edge_attr = edge_attr
    return data

def permute_edges(data, p: Optional[torch.Tensor] = None):
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * 0.2)
    if permute_num == 0:
        permute_num = 1

    edge_index = data.edge_index
    edge_attr = data.edge_attr
    
    if isinstance(p, torch.Tensor):
        idx_drop = torch.multinomial(p, num_samples=permute_num, replacement=False)
    else:
        idx_drop = torch.multinomial(torch.softmax(torch.ones([edge_num], device=edge_index.device), dim=0), num_samples=permute_num, replacement=False)

    idx_keep = torch.arange(edge_num, device=edge_index.device)
    idx_keep = idx_keep[~torch.isin(idx_keep, idx_drop)]
    edge_index = edge_index[:, idx_keep]
    edge_attr = edge_attr[idx_keep]
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    return data

def sub_graph(data, p: Optional[torch.Tensor] = None):
    # return data
    x = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    sub_num = int(data.num_nodes * (1-0.2))
    if isinstance(p, torch.Tensor):
        last_idx = torch.multinomial(p, num_samples=1, replacement=False)
    else:
        last_idx = torch.randint(0, data.num_nodes, (1, ), device=edge_index.device).to(edge_index.device)

    keep_idx = None
    diff = None
    for k in range(1, sub_num):
        keep_idx, _, _, _ = k_hop_subgraph(last_idx, 10, edge_index)
        # print("subgraph: {}, keep_idx size: {}".format(k, keep_idx.shape[0]) )
        if keep_idx.shape[0] == last_idx.shape[0] or keep_idx.shape[0] >= sub_num or k == sub_num - 1:
            combined = torch.cat((last_idx, keep_idx)).to(edge_index.device)
            uniques, counts = combined.unique(return_counts=True)
            diff = uniques[counts == 1]
            break

        last_idx = keep_idx

    diff_keep_num = min(sub_num - last_idx.shape[0], diff.shape[0])
    diff_keep_idx = torch.randperm(diff.shape[0])[:diff_keep_num].to(edge_index.device)
    final_keep_idx = torch.cat((last_idx, diff_keep_idx))
    
    drop_idx = torch.ones(x.shape[0], dtype=bool)
    drop_idx[final_keep_idx] = False
    x[drop_idx] = 0
        
    edge_index, edge_attr = subgraph(final_keep_idx, edge_index, edge_attr)
    
    data.x = x
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    return data

def mask_nodes(data, p: Optional[torch.Tensor] = None):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * 0.2)
    if mask_num == 0:
        mask_num = 1

    token = data.x.mean(dim=0)
    if isinstance(p, torch.Tensor):
        idx_mask = torch.multinomial(p, num_samples=mask_num, replacement=False)
    else:
        idx_mask = torch.multinomial(torch.softmax(torch.ones([node_num], device=data.x.device), dim=0), num_samples=mask_num, replacement=False)

    # data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32, device=data.x.device)
    data.x[idx_mask] = token.clone().detach()
    return data


if __name__=='__main__':
    x = torch.rand([1155, 1024])
    adj = torch.rand([2, 402794])
