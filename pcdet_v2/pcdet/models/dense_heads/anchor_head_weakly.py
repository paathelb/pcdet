import numpy as np
import torch.nn as nn
import torch
from .anchor_head_single import AnchorHeadSingle

from pcdet.utils.box_utils import boxes3d_lidar_to_kitti_camera_nocopy
from pcdet.utils.geo_trans_torch import GeoTransTorch
import torch.nn.functional as F
from .target_assigner.weakly2d_3d_target_assigner import Weakly2D3DTargetAssigner

class AnchorHeadWeakly(AnchorHeadSingle):
    def __init__(self,  model_cfg, input_channels,
                    num_class, class_names, grid_size, 
                    point_cloud_range, predict_boxes_when_training=True,
                    loss_components=["3d_cls", "3d_reg"], **kwargs):
        super().__init__(
            model_cfg=model_cfg, input_channels=input_channels, 
            num_class=num_class, class_names=class_names, grid_size=grid_size, 
            point_cloud_range=point_cloud_range, 
            predict_boxes_when_training=predict_boxes_when_training,)
        
        self.loss_components = loss_components

    
    def get_loss(self):
        tb_dict = {}
        
        final_loss = torch.zeros(1).cuda()
        if "3d_cls" in self.loss_components:
            cls_loss, tb_cls_dict = self.get_cls_layer_loss()
            tb_dict.update(tb_cls_dict)
            final_loss += cls_loss
        if "3d_reg" in self.loss_components:
            box_loss, tb_dict_box = self.get_box_reg_layer_loss()
            tb_dict.update(tb_dict_box)
            final_loss += box_loss
        if "2d_reg" in self.loss_components:
            box2d_loss, tb_dict_box2d = self.get_box_2d_reg_layer_loss()
            tb_dict.update(tb_dict_box2d)
            final_loss += box2d_loss

        tb_dict["final_loss"] = final_loss.item()
        return final_loss, tb_dict


    def get_box_2d_reg_layer_loss(self):
        # TODO Add the comments
        pred_boxes3d = self.forward_ret_dict["batch_box_preds"]         # B x N x 7 # TODO Why don't we use self.forward_ret_dict['box_preds']?sss
        box_cls_labels = self.forward_ret_dict["box_cls_labels"]        # B x N
        batch_size = int(pred_boxes3d.shape[0])

        lidar2cam = self.forward_ret_dict["trans_lidar_to_cam"]         # B x 4 x 4
        cam2img = self.forward_ret_dict["trans_cam_to_img"]             # B x 3 x 4
        # lidar2cam = [torch.tensor(lidar2cam[idx]) for idx in range(batch_size)]
        # cam2img = [torch.tensor(cam2img[idx]) for idx in range(batch_size)]

        lidar2cam = [lidar2cam[idx].clone().detach() for idx in range(batch_size)]  # List len B
        cam2img = [cam2img[idx].clone().detach() for idx in range(batch_size)]      # List len B

        # V2C = [torch.tensor(calib[idx].V2C) for idx in range(batch_size)]
        # R0 = [torch.tensor(calib[idx].R0) for idx in range(batch_size)]

        # Convert to CUDA
        lidar2cam = torch.stack(lidar2cam, dim=0).to(pred_boxes3d.device)   # B x 4 x 4
        cam2img = torch.stack(cam2img, dim=0).to(pred_boxes3d.device)       # B x 3 x 4
        # Convert from lidar reference frame to camera frame
        pred_bboxes_camera = torch.stack([boxes3d_lidar_to_kitti_camera_nocopy(pred_boxes3d[idx], lidar2cam[idx])
                                          for idx in range(batch_size)], dim=0)     # B x N x 7

        # Encode 2D box from 3D box prediction
        cam2img = cam2img.unsqueeze(1).expand(-1, pred_boxes3d.shape[1], -1, -1)    # B x 1 x N x 3 x 4
        rotys = pred_bboxes_camera[:,:,-1].reshape(-1)                              # B*N
        dims =  pred_bboxes_camera[:,:,3:6].reshape(-1,3)                           # B*N x 3
        locs = pred_bboxes_camera[:,:,:3].reshape(-1,3)                             # B*N x 3
        img_size = torch.max(self.forward_ret_dict['image_shape'], dim=0)[0]        # TODO Why use the max here?
        pred_boxes2d = GeoTransTorch.encode_box2d(                                  # B*N x 4
                       rotys, dims, locs, cam2img.reshape(-1,3,4), img_size=torch.flip(img_size, dims=[0]))

        positives = box_cls_labels > 0          # B x N
        positives = positives.reshape(-1)       # B*N
        pred_boxes2d = pred_boxes2d[positives]  # P x 4
        targets_2d = self.forward_ret_dict["box2d_reg_targets"].reshape(-1,4)[positives]    # P x 4

        loss = F.smooth_l1_loss(pred_boxes2d, targets_2d)                                   # TODO Why use smooth l1 loss here?
        
        loss_dict = {}
        loss_dict["weakly_2d_3d_reg_loss"] = loss.item()
        return loss, loss_dict

    # def assign_targets(self, gt_boxes, gt_boxes2d=None,
    #                   trans_lidar_to_cam=None, trans_cam_to_img=None,
    #                   points=None, image_shape=None):
    #     """
    #     Args:
    #         gt_boxes: (B, M, 8)
    #     Returns:
    #     """
    #     # if gt_boxes
    #     if True:
    #         targets_dict = self.target_assigner.assign_targets(
    #             self.anchors, gt_boxes, gt_boxes2d
    #         )
    #     else:
    #         target_dict = self.target_assigner.assign_targets(
    #             self.anchors, gt_bboxes2d, trans_lidar2cam,
    #             trans_cam2img, points, image_shape,)
    #     return targets_dict

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d'] # B x 384 x ny x nx 
        # Use the spatial_features_2d for both classification and box regression
        cls_preds = self.conv_cls(spatial_features_2d) # B x 2 x ny x nx 
        box_preds = self.conv_box(spatial_features_2d) # B x 14 x ny x nx 

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] # B x ny x nx x 2
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] # B x ny x nx x 14

        self.forward_ret_dict['cls_preds'] = cls_preds # B x ny x nx x 2
        self.forward_ret_dict['box_preds'] = box_preds # B x ny x nx x 14

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d) # B x 4 x ny x nx
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous() # B x ny x nx x 4
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds # B x ny x nx x 4
        else:
            dir_cls_preds = None
        
        if self.training:
            if "gt_boxes2d" not in data_dict:
                targets_dict = self.assign_targets(
                    gt_boxes=data_dict['gt_boxes']
                )
            else:
                # Only support for axis aligned target assigner
                if isinstance(self.target_assigner, Weakly2D3DTargetAssigner):
                    targets_dict = self.assign_targets(
                        gt_boxes=data_dict["gt_boxes"],
                        gt_boxes2d=data_dict["gt_boxes2d"],
                        trans_lidar_to_cam=data_dict["trans_lidar_to_cam"],
                        trans_cam_to_img=data_dict["trans_cam_to_img"],
                        points=data_dict["points"],
                        image_shape=data_dict["image_shape"],
                        frame_id = data_dict["frame_id"],
                        batch_segpts = data_dict['segpts'],
                        segpts_cnt = data_dict['segpts_cnt']
                    ) # Dict Keys: box_cls_labels: 4 x num_anchors, box2d_cls_labels 4 x num_anchors, 
                      #            box2d_reg_targets: 4 x num_anchors x 4, reg2d_weights: 4 x numanchors
                else:
                    targets_dict = self.assign_targets(
                        gt_boxes=data_dict["gt_boxes"],
                        gt_boxes2d=data_dict["gt_boxes2d"],
                    )
            self.forward_ret_dict.update(targets_dict)
            self.forward_ret_dict['trans_lidar_to_cam'] = data_dict['trans_lidar_to_cam']   # B x 4 x 4
            self.forward_ret_dict['trans_cam_to_img'] = data_dict['trans_cam_to_img']       # B x 3 x 4
            self.forward_ret_dict['gt_boxes'] = data_dict['gt_boxes']                       # B x N x 8
            self.forward_ret_dict['gt_boxes2d'] = data_dict['gt_boxes2d']                   # B x N x 4
            self.forward_ret_dict['image_shape'] = data_dict['image_shape']                 # B x 2

            # Why store this?
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            ) # B x num_anchors x 1, B x num_anchors x 7
            self.forward_ret_dict['batch_cls_preds'] = batch_cls_preds # B x num_anchors x 1
            self.forward_ret_dict['batch_box_preds'] = batch_box_preds # B x num_anchors x 7
            self.forward_ret_dict['cls_preds_normalized'] = False

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
