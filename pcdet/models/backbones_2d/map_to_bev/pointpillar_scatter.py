import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords'] # V x BEV, V x 4
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device) # BEV x T

            batch_mask = coords[:, 0] == batch_idx # V
            this_coords = coords[batch_mask, :] # Vsub x 4
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3] # Vsub 
            # TODO Know the intuition. Why sum these to get the indices?
            indices = indices.type(torch.long) # Vsub
            pillars = pillar_features[batch_mask, :] # Vsub x BEV
            pillars = pillars.t() # BEV x Vsub
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0) # 4 x BEV x T
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx) # 4 x BEV x ny x nx
        batch_dict['spatial_features'] = batch_spatial_features # 4 x BEV x ny x nx
        return batch_dict
