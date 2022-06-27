import numpy as np
import torch
import torch.nn.functional as F
import pcdet.models.dense_heads.target_assigner.kitti_utils_official as kitti_utils_official

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils

# change
from pcdet.utils.box_utils import boxes3d_lidar_to_kitti_camera_nocopy
from pcdet.utils.geo_trans_torch import GeoTransTorch
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
# from pcdet.models.dense_heads.fast_ground_removal.ground_removal import Processor
from pcdet.utils.box_utils import boxes3d_lidar_to_kitti_camera_nocopy
from pcdet.utils.box_utils import boxes_iou_normal
from pcdet.utils.box_utils import boxes3d_nearest_bev_iou

class Weakly2D3DTargetAssigner(object):
    def __init__(self, model_cfg, class_names, 
                        topk, rank_by_num_points, 
                        points_inside_2dbox_only):
        super().__init__()
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
        gt_boxes3d = gt_boxes
        
        for idx in range(batch_size):
            cur_gt = gt_boxes2d[idx]
            cur_gt_3d = gt_boxes3d[idx]

            if False:
                cnt = cur_gt_3d.__len__() - 1
                while cnt > 0 and cur_gt_3d[cnt].sum() == 0:
                    cnt -= 1
            else: 
                cnt = cur_gt.__len__() - 1
                while cnt > 0 and cur_gt[cnt].sum() == 0:
                    cnt -= 1

            cur_gt = cur_gt[:cnt + 1]
            cur_gt_3d = cur_gt_3d[:cnt + 1]
#             cur_gt_classes = gt_classes[k][:cnt + 1].int()
            cur_gt_classes = torch.tensor((cnt+1)*[1], device=cur_gt.device).int()

            target_list = []
            
            all_anchors2d = self.generate_anchors2d(
                all_anchors, trans_lidar_to_cam[idx], 
                trans_cam_to_img[idx], image_shape=image_shape[idx])     # TODO Double check the code
            
            points_mask = points[:,0]==idx
            points_idx = points[points_mask]
            # Only consider points inside GT 2D Bounding Boxes
            
            if self.points_inside_2dbox_only: # TODO Consider points that fall inside 2D box and at the same time 
                                              # these points form cluster where their distances are within a threshold
                points_idx = self.select_segmented_points(
                                                points_idx, 
                                                trans_lidar_to_cam[idx], 
                                                trans_cam_to_img[idx],
                                                cur_gt,
                                                image_shape[idx])
            
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
                    gt_boxes2d = cur_gt[mask]
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
                                 trans_cam_to_img=None, points=None, code_size=4, gt_boxes2d=None):
        
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]

        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1 # IDs of GT boxes that have huge IoU to the anchor

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            anchor_by_gt_overlap = boxes_iou_normal(anchors[:, 0:4], gt_boxes[:, 0:4]) #Intersection of the 2D anchors and the 2D ground truth boxes
            #anchor_by_gt_overlap = boxes3d_nearest_bev_iou(anchors3d[:, 0:7], gt_boxes[:, 0:7])

            anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().detach().numpy().argmax(axis=1)).cuda()
            anchor_to_gt_max = anchor_by_gt_overlap[torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax]

            gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().detach().numpy().argmax(axis=0)).cuda()
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)]
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1

            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0] # What anchors have the max overlap to each of the gt boxes
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap] # What gt indices correspond to these anchors with max overlap
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int()

            pos_inds = anchor_to_gt_max >= matched_threshold #what anchors have IoU >= matched threshold
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds] #what is the GT index corresponding to those anchors with IoU >= matched threshold
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]
            gt_ids[pos_inds] = gt_inds_over_thresh.int()
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]
            
            index = torch.arange(num_anchors)
            for gt_idx in range(num_gt): # sort anchors per ground truth box # TODO consider only points corresponding to the 2D GT box, not the total filtered points for all GT box
                pos_anchors_mask =  gt_ids == gt_idx

                pos_anchors_iou = anchor_by_gt_overlap[pos_anchors_mask, gt_idx]

                pos_anchors3d = anchors3d[pos_anchors_mask]

                _, sort_rank = pos_anchors_iou.sort(descending=True)

                pos_anchors3d = pos_anchors3d[sort_rank]

                with torch.no_grad(): # Why put here?
                    if len(points) > 0:
                        num_points_in_anchors = \
                                [(points_in_boxes_gpu(points[:,1:4].unsqueeze(0), pos_anchor3d.reshape(1,1,7))>=0).sum().item()
                                                    for pos_anchor3d in pos_anchors3d]
                        topk = min(len(num_points_in_anchors), self.topk)
                        # The top k only consider number of points 
                        # Ignore anchor size since all of anchors have the same size
                        # TODO consider density 
                    else:
                        topk == 0

                    if topk == 0:
                        continue

                    _, topk_idx = torch.topk(torch.tensor(num_points_in_anchors), topk) # NOTE In the case of multiple same num_points, selection seems random.
                    # It doesn't choose index with higher IOU. 

                pos_idx = index[pos_anchors_mask][sort_rank][topk_idx] #remove [topk_idx] and put [:topk] if 2D IoU is the only basis for ranking 
                gt_ids[pos_anchors_mask] = -1
                gt_ids[pos_idx] = gt_idx
                labels[pos_anchors_mask] = -1
                labels[pos_idx] = gt_classes[gt_idx]

            labels[bg_inds] = 0
            # do the topk filtering
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)
        
        bbox2d_targets = anchors.new_zeros((num_anchors, 4))

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            #bbox2d_targets = gt_boxes2d[gt_ids.long(), :]
            bbox2d_targets = gt_boxes[gt_ids.long(), :]
#             bbox2d_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes2d, fg_anchors)

        reg_weights = anchors.new_zeros((num_anchors,))

#         if self.norm_by_num_examples:
#             num_examples = (labels >= 0).sum()
#             num_examples = num_examples if num_examples > 1.0 else 1.0
#             reg_weights[labels > 0] = 1.0 / num_examples
#         else:

        reg_weights[labels > 0] = 1.0 

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


    def select_segmented_points(self, points, tran_lidar_to_cam, tran_cam_to_img, gt_2d, image_shape=None):

        # points3d = torch.matmul(points[:, 1:5], tran_lidar_to_cam.T)
        # points2d = torch.matmul(points3d, tran_cam_to_img.T)

        # depth = points2d[..., 2:3]
        # points2d = points2d[..., :2] / depth    

        # # Remove points outside ground truth bounding boxes
        # final = points.new_zeros((points2d.shape[0],), dtype=bool)
        # for box in gt_2d:
        #     inds = points2d[:, 0] >= box[0]
        #     inds = torch.logical_and(inds, points2d[:, 0] < box[2])
        #     inds = torch.logical_and(indsj, points2d[:, 1] > box[1])
        #     inds = torch.logical_and(inds, points2d[:, 1] < box[3])
        #     inds = torch.logical_and(inds, depth.reshape(-1j) >= 0)
        #     final = torch.logical_or(inds, final)
        
        # final = torch.from_numpy(final)
        
        pc_all, object_filter_all = kitti_utils_official.get_point_cloud_weakly_version(points, tran_lidar_to_cam, tran_cam_to_img, image_shape, back_cut=False)
        mask_ground_all, ground_sample_points = kitti_utils_official.calculate_ground_weakly(pc_all, tran_lidar_to_cam, tran_cam_to_img, image_shape, 0.2, back_cut=False)
        
        z_list = [] # list of medians of depth of points inside 2D bbox
        index_list = [] # list of all objects indices
        valid_list = [] # list of valid objects. If pc in 3D box is >= 30, then it is valid.

        total_object_number = 0
        for i in range(len(gt_2d)):
            total_object_number += 1
            flag = 1

            _, object_filter = kitti_utils_official.get_point_cloud_weakly_version(points, tran_lidar_to_cam, tran_cam_to_img, image_shape, [gt_2d[i]], back_cut=False)
            pc = pc_all[object_filter == 1]
            
            valid_list.append(i)
            z_list.append(torch.median(pc[:, 2]))
            index_list.append(i)

        sort = torch.argsort(torch.tensor(z_list))
        object_list = list(torch.tensor(index_list)[sort])

        mask_object = torch.ones((pc_all.shape[0])).float().to(pc_all.device)
        mask_seg_best_final = torch.zeros((pc_all.shape[0])).float().to(pc_all.device)
        thresh_seg_max = 7

        for i in object_list:
            result = torch.zeros((7, 2)).float().to(pc_all.device)
            count = 0
            mask_seg_list = []
            
            for j in range(thresh_seg_max):
                thresh = (j + 1) * 0.1
                _, object_filter = kitti_utils_official.get_point_cloud_weakly_version(
                                   points, tran_lidar_to_cam, tran_cam_to_img, image_shape, [gt_2d[i]], back_cut=False)
                
                filter_z = pc_all[:, 2] > 0
                mask_search = mask_ground_all* object_filter_all * mask_object * filter_z
                mask_origin = mask_ground_all * object_filter * mask_object * filter_z
                
                mask_seg = kitti_utils_official.region_grow_weakly_version(pc_all.detach().clone(), 
                                                                    mask_search, mask_origin, thresh, ratio=0.90)
                if mask_seg.sum() == 0:
                    continue

                if j >= 1:
                    mask_seg_old = mask_seg_list[-1] 
                    if mask_seg_old.sum() != (mask_seg * mask_seg_old).sum():
                        count += 1
                result[count, 0] = j  
                result[count, 1] = mask_seg.sum()
                mask_seg_list.append(mask_seg)

            best_j = result[torch.argmax(result[:, 1]), 0].item()

            try:
                mask_seg_best = mask_seg_list[int(best_j)]
                mask_object *= (1 - mask_seg_best)
                pc = pc_all[mask_seg_best == 1].detach().clone()

            except IndexError:
                # print("bad region grow result! deprecated")
                continue

            if i not in valid_list: # Why do we have access to the valid list?
                continue
            
            if kitti_utils_official.check_truncate(image_shape, gt_2d[i]): # different with objects[i].boxes[0].box?
                # print('object %d truncates in %s, with bbox %s' % (i, seq, str(objects[i].boxes_origin[0].box)))

                mask_origin_new = mask_seg_best
                mask_search_new = mask_ground_all
                thresh_new      = (best_j + 1) * 0.1

                mask_seg_for_truncate = kitti_utils_official.region_grow_weakly_version(pc_all.detach().clone(),
                                                                                    mask_search_new,
                                                                                    mask_origin_new,
                                                                                    thresh_new,
                                                                                    ratio=None)
                pc_truncate = pc_all[mask_seg_for_truncate == 1].detach().clone()
                mask_seg_best_final = torch.logical_or(mask_seg_best_final, mask_seg_for_truncate)

            else:
                mask_seg_best_final = torch.logical_or(mask_seg_best_final, mask_seg_best)

        return points[mask_seg_best_final]