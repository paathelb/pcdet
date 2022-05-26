import numpy as np
import torch
import torch.nn.functional as F

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils

# change
from pcdet.utils.box_utils import boxes3d_lidar_to_kitti_camera_nocopy
from pcdet.utils.geo_trans_torch import GeoTransTorch
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
# from pcdet.models.dense_heads.fast_ground_removal.ground_removal import Processor
from pcdet.utils.box_utils import boxes3d_lidar_to_kitti_camera_nocopy
from pcdet.utils.box_utils import boxes_iou_normal

class Weakly2D3DTargetAssigner(object):
    def __init__(self, model_cfg, class_names, 
                        topk, rank_by_num_points, 
                        points_inside_2dbox_only):
        super().__init__()
        # import pdb; pdb.set_trace(0)
        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG
        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG
        self.class_names = np.array(class_names)
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        self.anchor_target_cfg = anchor_target_cfg
        self.topk = topk
        self.rank_by_num_points = rank_by_num_points
        self.points_inside_2dbox_only=points_inside_2dbox_only
        
        self.matched_thresholds = {}
        self.unmatched_thresholds = {}
        for config in anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']
            # self.iou_weight = config['iou_weight']
            # self.num_points_weight = config['num_points_weight']

        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False)


    def assign_targets(self, all_anchors, gt_boxes, gt_boxes2d_with_classes,
                         trans_lidar_to_cam, trans_cam_to_img, points, image_shape):
        """
        Args:          
            gt_boxes2d_with_classes: (B, M, 8)
            calibs
            points
            image_shape
        Returns:

        """
        
        bbox2d_targets = []
        cls_labels = []
        reg_weights = []

        batch_size = gt_boxes2d_with_classes.shape[0]
        # Check if there are gt classes
        gt_classes = gt_boxes2d_with_classes[:,:,-1]
        gt_boxes2d = gt_boxes2d_with_classes
        
        for idx in range(batch_size):
            cur_gt = gt_boxes2d[idx]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1

            cur_gt = cur_gt[:cnt + 1]
#             cur_gt_classes = gt_classes[k][:cnt + 1].int()
            cur_gt_classes = torch.tensor((cnt+1)*[1], device=cur_gt.device).int()

            target_list = []
            
            all_anchors2d = self.generate_anchors2d(
                all_anchors, trans_lidar_to_cam[idx], 
                trans_cam_to_img[idx], image_shape=image_shape[idx])     
            
            
            points_mask = points[:,0]==idx
            points_idx = points[points_mask]
            # Only consider points inside GT 2D Bounding Boxes

            if self.points_inside_2dbox_only:
                points_idx = self.select_only_points_inside_2dbbox(
                                                points_idx, 
                                                trans_lidar_to_cam[idx], 
                                                trans_cam_to_img[idx],
                                                cur_gt)
            
            for anchor_class_name, anchors, anchors3d in zip(self.anchor_class_names, all_anchors2d, all_anchors):
                if cur_gt_classes.shape[0] > 1:
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name)
                else:
                    mask = torch.tensor([self.class_names[c - 1] == anchor_class_name
                                        for c in cur_gt_classes], dtype=torch.bool)
                if self.use_multihead:
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                    selected_classes = cur_gt_classes[mask]
                else:
                    feature_map_size = anchors.shape[:3]
                    anchors = anchors.view(-1, anchors.shape[-1])
                    anchors3d = anchors3d.view(-1, anchors3d.shape[-1])
                    selected_classes = cur_gt_classes[mask]
                
                single_target = self.assign_targets_single(
                    anchors,
                    anchors3d,
                    cur_gt[mask],
                    gt_classes=selected_classes,
                    matched_threshold=self.matched_thresholds[anchor_class_name],
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name],
                    trans_lidar_to_cam=trans_lidar_to_cam[idx],
                    trans_cam_to_img=trans_cam_to_img[idx],
                    points=points_idx,
                )
                target_list.append(single_target)

            if self.use_multihead:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list],
                    'box2d_reg_targets': [t['box2d_reg_targets'].view(-1, 4) for t in target_list], # 4 for 2d box
                    'reg_weights': [t['reg_weights'].view(-1) for t in target_list]
                }        
                target_dict['box2d_reg_targets'] = torch.cat(target_dict['box2d_reg_targets'], dim=0)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)
            else:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(*feature_map_size, -1) for t in target_list],
                    'box2d_reg_targets': [t['box2d_reg_targets'].view(*feature_map_size, -1, 4) for t in target_list], #4 for 2d box
                    'reg_weights': [t['reg_weights'].view(*feature_map_size, -1) for t in target_list]
                }
                target_dict['box2d_reg_targets'] = torch.cat(target_dict['box2d_reg_targets'], dim=-2).view(-1, 4) #4 for 2d box
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(-1)

            bbox2d_targets.append(target_dict['box2d_reg_targets'])
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])

        bbox2d_targets = torch.stack(bbox2d_targets, dim=0)
        cls_labels = torch.stack(cls_labels, dim=0)
        reg_weights = torch.stack(reg_weights, dim=0)
        
        all_targets_dict = {
            'box_cls_labels': cls_labels,
            'box2d_cls_labels': cls_labels,
            'box2d_reg_targets': bbox2d_targets,
            'reg2d_weights': reg_weights
        }
            
        return all_targets_dict


    def assign_targets_single(self, anchors, anchors3d, gt_boxes,
                                 gt_classes, matched_threshold=0.60,
                                 unmatched_threshold=0.45, trans_lidar_to_cam=None,
                                 trans_cam_to_img=None, points=None, code_size=4):
        
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]

        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1 # IDs of GT boxes that has huge IoU to the anchor

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            anchor_by_gt_overlap = boxes_iou_normal(anchors[:, 0:4], gt_boxes[:, 0:4]) #Intersection of the 2D anchors and the 2D ground truth boxes

            anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().detach().numpy().argmax(axis=1)).cuda()
            anchor_to_gt_max = anchor_by_gt_overlap[torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax]

            gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().detach().numpy().argmax(axis=0)).cuda()
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)]
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1

            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0] #what anchors have the max overlap to each of the gt boxes
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap] #what gt indices correspond to these anchors with max overlap
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int()

            pos_inds = anchor_to_gt_max >= matched_threshold #what anchors have IoU >= matched threshold
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds] #what is the GT index corresponding to those anchors with IoU >= matched threshold
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]
            gt_ids[pos_inds] = gt_inds_over_thresh.int()
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]

            index = torch.arange(num_anchors)
            for gt_idx in range(num_gt): # sort anchors per ground truth box
                pos_anchors_mask =  gt_ids == gt_idx

                pos_anchors_iou = anchor_by_gt_overlap[pos_anchors_mask, gt_idx]

                pos_anchors3d = anchors3d[pos_anchors_mask]
                sort_pos_anchors_iou, sort_rank = pos_anchors_iou.sort(descending=True)

                pos_anchors3d = pos_anchors3d[sort_rank]
                with torch.no_grad(): #Why put here?
                    num_points_in_anchors = \
                            [(points_in_boxes_gpu(points[:,1:4].unsqueeze(0), pos_anchor3d.reshape(1,1,7))>=0).sum().item()
                                                 for pos_anchor3d in pos_anchors3d]
                    # the top k only consider number of points # ignore anchor size since all of anchors have the same size
                    # TODO modify to density 
                    topk = min(len(num_points_in_anchors), self.topk)
                    if topk == 0:
                        continue
                    topk_num, topk_idx = torch.topk(torch.tensor(num_points_in_anchors), topk)
                    # set xxx as zero
                pos_idx = index[pos_anchors_mask][sort_rank][topk_idx]
                gt_ids[pos_anchors_mask] = -1
                gt_ids[pos_idx] = gt_idx
                #labels[pos_anchors_mask] = -1
                labels[pos_idx] = gt_classes[gt_idx]

            # do the topk filtering
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)
        
        bbox2d_targets = anchors.new_zeros((num_anchors, 4))
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            bbox2d_targets = gt_boxes[gt_ids.long(), :]
#             bbox2d_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes2d, fg_anchors)
        
        reg_weights = anchors.new_zeros((num_anchors,))

#         if self.norm_by_num_examples:
#             num_examples = (labels >= 0).sum()
#             num_examples = num_examples if num_examples > 1.0 else 1.0
#             reg_weights[labels > 0] = 1.0 / num_examples
#         else:
        #reg_weights[labels > 0] = 1.0 # TODO Change the way this is defined to align with the topk anchors
        reg_weights[gt_ids != -1] = 1.0
        #import pdb; pdb.set_trace() 
        ret_dict = {
            'box_cls_labels': labels,
            'box2d_reg_targets': bbox2d_targets,
            'reg_weights': reg_weights,
        }

        return ret_dict

    def remove_ground_points_(self, batch_points):
        # Code from https://github.com/SilvesterHsu/LiDAR_ground_removal
        batch_points_np = batch_points.cpu().numpy()
        batch_points_np = batch_points_np * np.array([1,1,-1], dtype=np.float32)

        process = Processor(n_segments=70, n_bins=80, line_search_angle=0.3, max_dist_to_line=0.15,
                            sensor_height=1.73, max_start_height=0.5, long_threshold=8)
        batch_non_ground = process(batch_points_np)
     
        return torch.from_numpy(batch_non_ground).float().cuda()


    def generate_anchors2d(self, anchors, trans_lidar_to_cam, trans_cam_to_img, image_shape):
        anchors = torch.stack(anchors, dim=0)
        #import pdb; pdb.set_trace()
        N, X, Y, Z, level, anchor_type, code_size = anchors.shape 
        anchors = anchors.reshape(-1, code_size)
        
        # Convert from lidar 3D coordinates to camera 3D coordinates
        # V2C = [torch.tensor(calib[idx].V2C) for idx in range(batch_size)]
        # R0 = [torch.tensor(calib[idx].R0) for idx in range(batch_size)]
        # 

        # Convert to cuda
        # V2C = torch.stack(V2C, dim=0).to(anchors.device)
        # R0 = torch.stack(R0, dim=0).to(anchors.device)
        anchors_camera = boxes3d_lidar_to_kitti_camera_nocopy(anchors, trans_lidar_to_cam)

        w = image_shape[1].item()
        h = image_shape[0].item()
        
        # trans_cam_to_img = torch.tensor(trans_cam_to_img[index]).reshape(1, 3, 4)
        roty = anchors_camera[:,6].reshape(1,-1)
        dim = anchors_camera[:,3:6]
        loc = anchors_camera[:,0:3]
        trans_cam_to_img = trans_cam_to_img.reshape(1,3,4)
        anchors_2d = GeoTransTorch.encode_box2d(roty, dim, loc, trans_cam_to_img, (w, h)) 
        anchors_2d = torch.unsqueeze(anchors_2d, dim=0)

        anchors_2d = anchors_2d.reshape(N, X, Y, Z, level, anchor_type, 4)
        anchors_2d = [i for i in anchors_2d]
        return anchors_2d


    def select_only_points_inside_2dbbox(self, points, tran_lidar_to_cam, tran_cam_to_img, gt_2d):

        points3d = torch.matmul(points[:, 1:5], tran_lidar_to_cam.T)
        points2d = torch.matmul(points3d, tran_cam_to_img.T)

        depth = points2d[..., 2:3]
        points2d = points2d[..., :2] / depth    


        # Remove points outside ground truth bounding boxes
        final = points.new_zeros((points2d.shape[0],), dtype=bool)
        for box in gt_2d:
            inds = points2d[:, 0] > box[0]
            inds = torch.logical_and(inds, points2d[:, 0] < box[2])
            inds = torch.logical_and(inds, points2d[:, 1] > box[1])
            inds = torch.logical_and(inds, points2d[:, 1] < box[3])
            inds = torch.logical_and(inds, depth.reshape(-1) > 0)
            final = torch.logical_or(inds, final)
        
        # final = torch.from_numpy(final)
        return points[final]