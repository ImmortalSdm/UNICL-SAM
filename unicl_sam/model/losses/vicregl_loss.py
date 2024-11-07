import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

def batch_all_gather(x):
    x_list = GatherLayer.apply(x)
    return torch.cat(x_list, dim=0)

def gather_center(x):
    x = batch_all_gather(x)
    x = x - x.mean(dim=0)
    return x

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def neirest_neighbores(input_maps, candidate_maps, distances, num_matches):
    batch_size = input_maps.size(0)

    if num_matches is None or num_matches == -1:
        num_matches = input_maps.size(1)

    topk_values, topk_indices = distances.topk(k=1, largest=False)
    topk_values = topk_values.squeeze(-1)
    topk_indices = topk_indices.squeeze(-1)

    sorted_values, sorted_values_indices = torch.sort(topk_values, dim=1)
    sorted_indices, sorted_indices_indices = torch.sort(sorted_values_indices, dim=1)

    mask = torch.stack(
        [
            torch.where(sorted_indices_indices[i] < num_matches, True, False)
            for i in range(batch_size)
        ]
    )
    topk_indices_selected = topk_indices.masked_select(mask)
    topk_indices_selected = topk_indices_selected.reshape(batch_size, num_matches)

    indices = (
        torch.arange(0, topk_values.size(1))
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to(topk_values.device)
    )
    indices_selected = indices.masked_select(mask)
    indices_selected = indices_selected.reshape(batch_size, num_matches)

    filtered_input_maps = batched_index_select(input_maps, 1, indices_selected)
    filtered_candidate_maps = batched_index_select(
        candidate_maps, 1, topk_indices_selected
    )

    return filtered_input_maps, filtered_candidate_maps


def neirest_neighbores_on_l2(input_maps, candidate_maps, num_matches):
    """
    input_maps: (B, H * W, C)
    candidate_maps: (B, H * W, C)
    """
    distances = torch.cdist(input_maps, candidate_maps)
    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)


def neirest_neighbores_on_location(
    input_location, candidate_location, input_maps, candidate_maps, num_matches
):
    """
    input_location: (B, H * W, 2)
    candidate_location: (B, H * W, 2)
    input_maps: (B, H * W, C)
    candidate_maps: (B, H * W, C)
    """
    distances = torch.cdist(input_location, candidate_location)
    return neirest_neighbores(input_maps, candidate_maps, distances, num_matches)


def exclude_bias_and_norm(p):
    return p.ndim == 1

def _location_to_NxN_grid(location, N=7, flip=False):
    i, j, h, w, H, W = location
    size_h_case = h / N
    size_w_case = w / N
    half_size_h_case = size_h_case / 2
    half_size_w_case = size_w_case / 2
    final_grid_x = torch.zeros(N, N)
    final_grid_y = torch.zeros(N, N)

    final_grid_x[0][0] = i + half_size_h_case
    final_grid_y[0][0] = j + half_size_w_case
    for k in range(1, N):
        final_grid_x[k][0] = final_grid_x[k - 1][0] + size_h_case
        final_grid_y[k][0] = final_grid_y[k - 1][0]
    for l in range(1, N):
        final_grid_x[0][l] = final_grid_x[0][l - 1]
        final_grid_y[0][l] = final_grid_y[0][l - 1] + size_w_case
    for k in range(1, N):
        for l in range(1, N):
            final_grid_x[k][l] = final_grid_x[k - 1][l] + size_h_case
            final_grid_y[k][l] = final_grid_y[k][l - 1] + size_w_case

    final_grid = torch.stack([final_grid_x, final_grid_y], dim=-1)
    if flip:
        # start_grid = final_grid.clone()
        for k in range(0, N):
            for l in range(0, N // 2):
                swap = final_grid[k, l].clone()
                final_grid[k, l] = final_grid[k, N - 1 - l]
                final_grid[k, N - 1 - l] = swap

    return final_grid

class VICRegL_Loss(nn.Module):
    def __init__(self, emb_dim=1024, alpha=0.75, l2_all_matches=1, inv_coeff=25.0, var_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.embedding_dim = emb_dim

        self.alpha = alpha
        self.l2_all_matches = l2_all_matches
        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff

        self.fast_vc_reg = False

    def _vicreg_loss(self, x, y):
        repr_loss = self.inv_coeff * F.mse_loss(x, y)

        x = gather_center(x)
        y = gather_center(y)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = self.var_coeff * (
            torch.mean(F.relu(1.0 - std_x)) / 2 + torch.mean(F.relu(1.0 - std_y)) / 2
        )

        x = x.permute((1, 0, 2))
        y = y.permute((1, 0, 2))

        *_, sample_size, num_channels = x.shape
        non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
        # Center features
        # centered.shape = NC
        x = x - x.mean(dim=-2, keepdim=True)
        y = y - y.mean(dim=-2, keepdim=True)

        cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
        cov_y = torch.einsum("...nc,...nd->...cd", y, y) / (sample_size - 1)
        cov_loss = (cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels) / 2 + (
            cov_y[..., non_diag_mask].pow(2).sum(-1) / num_channels
        ) / 2
        cov_loss = cov_loss.mean()
        cov_loss = self.cov_coeff * cov_loss

        return repr_loss, std_loss, cov_loss

    def _local_loss(
        self, maps_1, maps_2, location_1, location_2, with_loc_loss=False
    ):
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0

        # L2 distance based bacthing
        if self.l2_all_matches:
            num_matches_on_l2 = [None, None]
        else:
            num_matches_on_l2 = self.num_matches

        maps_1 = maps_1.flatten(0,1).flatten(-2).permute(0,2,1)
        maps_2 = maps_2.flatten(0,1).flatten(-2).permute(0,2,1)

        maps_1_filtered, maps_1_nn = neirest_neighbores_on_l2(
            maps_1, maps_2, num_matches=num_matches_on_l2[0]
        )
        maps_2_filtered, maps_2_nn = neirest_neighbores_on_l2(
            maps_2, maps_1, num_matches=num_matches_on_l2[1]
        )

        if self.fast_vc_reg:
            inv_loss_1 = F.mse_loss(maps_1_filtered, maps_1_nn, reduction='mean')
            inv_loss_2 = F.mse_loss(maps_2_filtered, maps_2_nn)
        else:
            inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2, var_loss_2, cov_loss_2 = self._vicreg_loss(maps_2_filtered, maps_2_nn)
            var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
            cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)

        inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)

        # Location based matching
        if with_loc_loss:
            location_1 = location_1.flatten(1, 2)
            location_2 = location_2.flatten(1, 2)

            maps_1_filtered, maps_1_nn = neirest_neighbores_on_location(
                location_1,
                location_2,
                maps_1,
                maps_2,
                num_matches=self.num_matches[0],
            )
            maps_2_filtered, maps_2_nn = neirest_neighbores_on_location(
                location_2,
                location_1,
                maps_2,
                maps_1,
                num_matches=self.num_matches[1],
            )

            if self.fast_vc_reg:
                inv_loss_1 = F.mse_loss(maps_1_filtered, maps_1_nn)
                inv_loss_2 = F.mse_loss(maps_2_filtered, maps_2_nn)
            else:
                inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(maps_1_filtered, maps_1_nn)
                inv_loss_2, var_loss_2, cov_loss_2 = self._vicreg_loss(maps_2_filtered, maps_2_nn)
                var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
                cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)

            inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)

        return inv_loss, var_loss, cov_loss

    def local_loss(self, embedding, glob_embedding, locations=[None, None]):
        num_views = len(embedding)
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                inv_loss_this, var_loss_this, cov_loss_this = self._local_loss(
                    embedding[i], embedding[j], locations[i], locations[j],
                )
                inv_loss = inv_loss + inv_loss_this
                var_loss = var_loss + var_loss_this
                cov_loss = cov_loss + cov_loss_this
                iter_ += 1

        if self.fast_vc_reg:
            inv_loss = self.inv_coeff * inv_loss / iter_
            var_loss = 0.0
            cov_loss = 0.0
            iter_ = 0
            for i in range(num_views):
                x = glob_embedding[i].flatten(0,1).flatten(-2).permute(0,2,1)
                std_x = torch.sqrt(x.var(dim=0) + 0.0001)
                var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
                x = x.permute(1, 0, 2)
                *_, sample_size, num_channels = x.shape
                non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
                x = x - x.mean(dim=-2, keepdim=True)
                cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
                cov_l = cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels
                cov_loss = cov_loss + cov_l.mean()
                iter_ = iter_ + 1
            var_loss = self.var_coeff * var_loss / iter_
            cov_loss = self.cov_coeff * cov_loss / iter_
        else:
            inv_loss = inv_loss / iter_
            var_loss = var_loss / iter_
            cov_loss = cov_loss / iter_

        return inv_loss, var_loss, cov_loss

    def global_loss(self, embedding, glob_embedding, maps=False):
        num_views = len(embedding)
        inv_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                inv_loss = inv_loss + F.mse_loss(embedding[i], embedding[j])
                iter_ = iter_ + 1
        inv_loss = self.inv_coeff * inv_loss / iter_

        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(num_views):
            x = glob_embedding[i] # B*node, seq, c
            x = x.flatten(0,1).flatten(-2).permute(0,2,1).mean(-2).contiguous()
            # x = embedding[i] # B*node, seq, c
            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
            cov_x = (x.T @ x) / (x.size(0) - 1)
            cov_loss = cov_loss + off_diagonal(cov_x).pow_(2).sum().div(
                self.embedding_dim
            )
            iter_ = iter_ + 1
        var_loss = self.var_coeff * var_loss / iter_
        cov_loss = self.cov_coeff * cov_loss / iter_

        return inv_loss, var_loss, cov_loss

    def compute_metrics(self, inputs):
        def correlation_metric(x):
            x_centered = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-05)
            return torch.mean(
                off_diagonal((x_centered.T @ x_centered) / (x.size(0) - 1))
            )

        def std_metric(x):
            x = F.normalize(x, p=2, dim=1)
            return torch.mean(x.std(dim=0))

        representation = batch_all_gather(inputs["representation"][0])
        corr = correlation_metric(representation)
        stdrepr = std_metric(representation)

        if self.alpha > 0.0:
            embedding = batch_all_gather(inputs["embedding"][0])
            core = correlation_metric(embedding)
            stdemb = std_metric(embedding)
            return dict(stdr=stdrepr, stde=stdemb, corr=corr, core=core)

        return dict(stdr=stdrepr, corr=corr)

    def forward(self, base_feat, aug_feat, is_val=False, with_local_loss=False):
        # with torch.no_grad():
        #     logs = self.compute_metrics(inputs)
        loss = 0.0

        # Global criterion
        if self.alpha > 0.0:
            # global_pool_feats = inputs.flatten(1,2).flatten(-2).permute(0,1,3,2).mean(-2).contiguous()
            glob_base_feat = gather_center(base_feat)
            glob_aug_feat = gather_center(aug_feat)
            inv_loss, var_loss, cov_loss = self.global_loss( # ["embedding"]
                (base_feat, aug_feat), (glob_base_feat, glob_aug_feat)
            )
            loss = loss + self.alpha * (inv_loss + var_loss + cov_loss)
            # logs.update(dict(inv_l=inv_loss, var_l=var_loss, cov_l=cov_loss,))

        # Local criterion
        # Maps shape: B, C, H, W
        # With convnext actual maps shape is: B, H * W, C
        if self.alpha < 1.0 and with_local_loss:
            (
                maps_inv_loss,
                maps_var_loss,
                maps_cov_loss,
            ) = self.local_loss(
                (base_feat, aug_feat), (glob_base_feat, glob_aug_feat)
            )
            loss = loss + (1 - self.alpha) * (
                maps_inv_loss + maps_var_loss + maps_cov_loss
            )
            # logs.update(
            #     dict(minv_l=maps_inv_loss, mvar_l=maps_var_loss, mcov_l=maps_cov_loss,)
            # )
        # import pdb; pdb.set_trace()

        return loss
    
