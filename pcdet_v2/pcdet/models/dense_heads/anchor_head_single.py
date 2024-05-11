import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate
from .target_assigner.weakly2d_3d_target_assigner import Weakly2D3DTargetAssigner

class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None
        if self.training:
            if "gt_boxes2d" not in data_dict:
                targets_dict = self.assign_targets(
                    gt_boxes=data_dict['gt_boxes']
                )
            else:
                # "gt_boxes2d"
                # only support for axis aligned target assigner
                if isinstance(self.target_assigner, Weakly2D3DTargetAssigner):
                    targets_dict = self.assign_targets(
                        gt_boxes=data_dict["gt_boxes"],
                        gt_boxes2d=data_dict["gt_boxes2d"],
                        trans_lidar_to_cam=data_dict["trans_lidar_to_cam"],
                        trans_cam_to_img=data_dict["trans_cam_to_img"],
                        points=data_dict["points"],
                        image_shape=data_dict["image_shape"],
                    )
                else:
                    targets_dict = self.assign_targets(
                        gt_boxes=data_dict["gt_boxes"],
                        gt_boxes2d=data_dict["gt_boxes2d"],
                    )
            self.forward_ret_dict.update(targets_dict)
            self.forward_ret_dict['trans_lidar_to_cam'] = data_dict['trans_lidar_to_cam']
            self.forward_ret_dict['trans_cam_to_img'] = data_dict['trans_cam_to_img']
            self.forward_ret_dict['gt_boxes'] = data_dict['gt_boxes']
            self.forward_ret_dict['gt_boxes2d'] = data_dict['gt_boxes2d']
            self.forward_ret_dict['image_shape'] = data_dict['image_shape']

            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            self.forward_ret_dict['batch_cls_preds'] = batch_cls_preds
            self.forward_ret_dict['batch_box_preds'] = batch_box_preds
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
