import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import pickle
import copy
from multiprocessing import Pool
import time

from pcdet.utils import calibration_kitti
import kitti_utils_official

def get_calib(root_split_path, idx):
    calib_file = root_split_path / 'calib' / ('%s.txt' % idx)
    assert calib_file.exists()
    return calibration_kitti.Calibration(calib_file)

def get_lidar(root_split_path, idx):
    lidar_file = root_split_path / 'velodyne' / ('%s.bin' % idx)
    assert lidar_file.exists()
    return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

def get_fov_flag(pts_rect, img_shape, calib):
    """
    Args:
        pts_rect:
        img_shape:
        calib:

    Returns:

    """
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag

def calib_to_matricies(calib):
    """
    Converts calibration object to transformation matricies
    Args:
        calib: calibration.Calibration, Calibration object
    Returns
        V2R: (4, 4), Lidar to rectified camera transformation matrix
        P2: (3, 4), Camera projection matrix
    """
    V2C = np.vstack((calib.V2C, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    R0 = np.hstack((calib.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
    R0 = np.vstack((R0, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    V2R = R0 @ V2C
    P2 = calib.P2
    return V2R, P2

def select_segmented_points(index, kitti_infos, root_split_path):
        # TODO Consider only 2D GT boxes of Cars only. Study the code of pcdet datasets
        # TODO Consider a faster method to estimate ground plane points
        
        info = copy.deepcopy(kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        image_shape = info['image']['image_shape']
        calib = get_calib(root_split_path, sample_idx)

        #print(sample_idx)

        if 'annos' in info:
            annos = info['annos']
            car_index = torch.from_numpy(np.where(annos['name'] == 'Car')[0])
            gt_2d = torch.from_numpy(annos["bbox"])[car_index]

        # POINTS
        FOV_POINTS_ONLY = True
        points = get_lidar(root_split_path, sample_idx)

        if FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = get_fov_flag(pts_rect, image_shape, calib)
            points = points[fov_flag]
            points = torch.from_numpy(points)
            points = F.pad(input=points, pad=(1, 0, 0, 0), mode='constant', value=index)
        
        tran_lidar_to_cam, tran_cam_to_img = calib_to_matricies(calib)
        tran_lidar_to_cam, trans_cam_to_img = torch.from_numpy(tran_lidar_to_cam), torch.from_numpy(tran_cam_to_img)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        points, tran_lidar_to_cam, tran_cam_to_img, gt_2d = \
            points.to(device), tran_lidar_to_cam.to(device), trans_cam_to_img.to(device), gt_2d.to(device)

        if len(gt_2d) == 0:
            print(str(sample_idx) + " is empty")
            with open('/home/hpaat/pcdet/data/kitti/segpts/' + str(sample_idx) + '.pkl', 'wb') as save_folder:
                save_dic = {'frame_id': sample_idx, 
                    'segpts': []}
                pickle.dump(save_dic , save_folder)
            return 

        pc_all, object_filter_all = kitti_utils_official.get_point_cloud_weakly_version(points, tran_lidar_to_cam, tran_cam_to_img, image_shape, back_cut=False)
        mask_ground_all, ground_sample_points = kitti_utils_official.calculate_ground_weakly(pc_all, tran_lidar_to_cam, tran_cam_to_img, image_shape, 0.2, back_cut=False)
        
        z_list = [] # list of medians of depth of points inside 2D bbox
        index_list = [] # list of all objects indices
        valid_list = [] # list of valid objects. If pc in 3D box is >= 30, then it is valid.

        total_object_number = 0
        object_filter_list = []
        for i in range(len(gt_2d)):
            total_object_number += 1
            flag = 1

            _, object_filter = kitti_utils_official.get_point_cloud_weakly_version(points, tran_lidar_to_cam, tran_cam_to_img, image_shape, [gt_2d[i]],  \
                                                                                   image_filter=object_filter_all, back_cut=False)
            object_filter_list.append(object_filter)
            pc = pc_all[object_filter == 1]
            #print(len(pc))
            if len(pc) > 10:
                valid_list.append(i)
                z_list.append(torch.median(pc[:, 2]))
                index_list.append(i)
            else: 
                continue

        sort = torch.argsort(torch.tensor(z_list))
        object_list = list(torch.tensor(index_list)[sort])

        mask_object = torch.ones((pc_all.shape[0])).float().to(pc_all.device)
        mask_seg_best_final = torch.zeros((pc_all.shape[0])).float().to(pc_all.device)
        thresh_seg_max = 7

        for i in object_list:
            result = torch.zeros((7, 2)).float().to(pc_all.device)
            count = 0
            mask_seg_list = []
            
            object_filter = object_filter_list[i]
            filter_z = pc_all[:, 2] > 0
            mask_search = mask_ground_all* object_filter_all * mask_object * filter_z
            mask_origin = mask_ground_all * object_filter * mask_object * filter_z
            for j in range(thresh_seg_max):
                thresh = (j + 1) * 0.1

                #_, object_filter = kitti_utils_official.get_point_cloud_weakly_version(
                #                   points, tran_lidar_to_cam, tran_cam_to_img, image_shape, [gt_2d[i]], back_cut=False)
                
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
 
        mask_seg_best_final = mask_seg_best_final > 0
        
        save_dic = {'frame_id': sample_idx, 
                    'segpts': points[mask_seg_best_final][:, 1:4]}

        with open('/home/hpaat/pcdet/data/kitti/segpts/' + str(sample_idx) + '.pkl', 'wb') as save_folder:
            pickle.dump(save_dic , save_folder) # Consider index=279 (pc index 000570) sometimes this returns empty, sometimes only one point. Why does the result change?

        print("Done processing: Point cloud index {}, {}/{}".format(sample_idx, index, len(kitti_infos))) 

if __name__ == "__main__":
    
    split = 'test' #'train'
    save_segmented = True

    root_path = Path("/home/hpaat/pcdet/data/kitti")
    root_split_path = root_path / ('training' if split != 'test' else 'testing')
    split_dir = root_path / 'ImageSets' / (split + '.txt')
    sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    kitti_infos = []

    for info_path in ['kitti_infos_' + split + '.pkl']:
        info_path = root_path / info_path
        if not info_path.exists():
            continue
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            kitti_infos.extend(infos)

    print('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    if save_segmented:
        pool = Pool(processes=8) # TODO Set processes as parameter

        start = time.time()
        for index in range(len(kitti_infos)):
            # TODO Make this faster
            segmented_points = select_segmented_points(index, kitti_infos, root_split_path)
            #pool.apply_async(select_segmented_points, (index, kitti_infos, root_split_path))

        pool.close()
        pool.join()
    
        print("runtime: %.4fs" % (time.time() - start))

    else:
        # Aggregate all info in one dic
        # NOTE Save as numpy array, not as torch tensor

        # import pdb; pdb.set_trace() 
        all_dic = {}
        for id in sample_id_list:
            with open('/home/hpaat/pcdet/data/kitti/segpts/' + str(id) + '.pkl', 'rb') as save_folder:
                try: 
                    save_dic = pickle.load(save_folder)
                    save_dic['segpts'] = save_dic['segpts'].detach().cpu().numpy()  
                except: save_dic = {'frame_id': id, 
                                    'segpts': np.array([], dtype="float32")}
                all_dic[save_dic['frame_id']] = save_dic['segpts']
                print(id)

        with open('/home/hpaat/pcdet/data/kitti/segpts/!all_pts.pkl', 'wb') as save_folder:
            pickle.dump(all_dic, save_folder)
