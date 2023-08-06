# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import tempfile
from os import path as osp

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log

from ..core import show_multi_modality_result, show_result
from ..core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose


@DATASETS.register_module()
class SuscapeDataset(Custom3DDataset):
    r"""SUSCape Dataset.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list, optional): The range of point cloud used to
            filter invalid predicted boxes.
            Default: [0, -40, -3, 70.4, 40, 0.0].
    """
    CLASSES = ('Car', 'Pedestrian', 'ScooterRider', 'Truck', 'Scooter',
                'Bicycle', 'Van', 'Bus', 'BicycleRider', 
                'Trimotorcycle')

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 with_velocity=False,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False,
                 img_info_prototype='mmcv',
                 pcd_limit_range=[0, -40, -3, 70.4, 40, 0.0],
                 multi_adj_frame_id_cfg=None,
                 with_future_frame=False,
                 with_future_pred=False,
                 load_adj_bbox=False,
                 ego_cam='front',
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

        assert self.modality is not None
        self.pcd_limit_range = pcd_limit_range

        self.with_velocity = with_velocity
        self.eval_version = eval_version

        self.img_info_prototype = img_info_prototype
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.ego_cam = ego_cam
        self.with_future_frame = with_future_frame
        self.with_future_pred = with_future_pred
        self.load_adj_bbox = load_adj_bbox

   
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - img_prefix (str): Prefix of image files.
                - img_info (dict): Image info.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos["data_list"][index]


        input_dict = dict(
            # sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            # timestamp=info['timestamp'] / 1e6,
        )
        if 'ann_infos' in info:
            input_dict['ann_infos'] = info['ann_infos']

        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(
                        cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.
                            shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                    ))

                if not self.test_mode:
                    annos = self.get_ann_info(index)
                    input_dict['ann_info'] = annos
            else:
                assert 'bevdet' in self.img_info_prototype
                input_dict.update(dict(curr=info))
                if '4d' in self.img_info_prototype:
                    info_adj_list = self.get_adj_info(info, index)
                    input_dict.update(dict(adjacent=info_adj_list))
        return input_dict


    def get_adj_info(self, info, index):
        info_adj_list = []
        if not self.with_future_frame:
            max_id = min(index+1, len(self.data_infos["data_list"])-1)
            for select_id in range(*self.multi_adj_frame_id_cfg):
                select_id = max(index - select_id, 0)
                if not self.data_infos["data_list"][select_id]['scene_token'] == info['scene_token']:
                    info_adj_list.append(info)
                else:
                    info_adj_list.append(self.data_infos["data_list"][select_id])
        else:
            num_adj = len(list(range(*self.multi_adj_frame_id_cfg)))
            len_future = num_adj // 2
            len_prev = num_adj - len_future
            max_id = min(index+len_future+1, len(self.data_infos["data_list"])-1)
            select_list = list(range(-len_prev, 0)) + list(range(1, len_future+1))
            for select_id in select_list:
                select_id = min(max(index + select_id, 0), len(self.data_infos["data_list"])-1)
                if not self.data_infos["data_list"][select_id]['scene_token'] == info[
                        'scene_token']:
                    info_adj_list.append(info)
                else:
                    info_adj_list.append(self.data_infos["data_list"][select_id]) 
        if self.with_future_pred:
            if not self.data_infos["data_list"][max_id]['scene_token'] == info[
                    'scene_token']:
                info_adj_list.append(info)
            else:
                info_adj_list.append(self.data_infos["data_list"][max_id]) 
        return info_adj_list


    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                    0, 1, 2 represent xxxxx respectively.
        """

        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos['data_list'][index]
        
        # print("get info index", index, info["lidar_path"])
        # should we remove some object types?
        
        gt_names = info['gt_names']
        gt_bboxes_3d = np.array(info['gt_boxes']).astype(np.float32)

        if gt_bboxes_3d.shape[0] == 0:
            print(index, gt_bboxes_3d.shape)
        # gt_bboxes = annos['bbox']
        gt_bboxes_3d = LiDARInstance3DBoxes(
                    gt_bboxes_3d,
                    box_dim=gt_bboxes_3d.shape[-1],
                    origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            # bboxes=gt_bboxes,
            # labels=gt_labels,
            gt_names=gt_names,
            # plane=plane_lidar,
            # difficulty=difficulty
            )
        return anns_results
    def evaluate(self,
                 results,
                 metric='kitti',
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'kitti'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str: float]: results of each evaluation metric
        """
        assert ('waymo' in metric or 'kitti' in metric), \
            f'invalid metric {metric}'
        if 'kitti' in metric:
            result_files, tmp_dir = self.format_results(
                results,
                pklfile_prefix,
                submission_prefix)
            from mmdet3d.core.evaluation import kitti_eval
            
            gt_annos = self.format_gt(self.data_infos['data_list'])

            if isinstance(result_files, dict):
                ap_dict = dict()
                for name, result_files_ in result_files.items():
                    eval_types = ['bev', '3d']
                    ap_result_str, ap_dict_ = kitti_eval(
                        gt_annos,
                        result_files_,
                        self.CLASSES,
                        eval_types=eval_types)
                    for ap_type, ap in ap_dict_.items():
                        ap_dict[f'{name}/{ap_type}'] = float(
                            '{:.4f}'.format(ap))

                    print_log(
                        f'Results of {name}:\n' + ap_result_str, logger=logger)

            else:
                ap_result_str, ap_dict = kitti_eval(
                    gt_annos,
                    result_files,
                    self.CLASSES,
                    eval_types=['bev', '3d'])
                print_log('\n' + ap_result_str, logger=logger)
        
        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return ap_dict

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if not isinstance(outputs[0], dict):
            result_files = self.bbox2result_kitti2d(outputs, self.CLASSES,
                                                    pklfile_prefix,
                                                    submission_prefix)
        elif 'pts_bbox' in outputs[0] or 'img_bbox' in outputs[0]:
            result_files = dict()
            for name in outputs[0]:
                results_ = [out[name] for out in outputs]
                pklfile_prefix_ = pklfile_prefix + name
                if submission_prefix is not None:
                    submission_prefix_ = submission_prefix + name
                else:
                    submission_prefix_ = None
                if 'img' in name:
                    result_files = self.bbox2result_kitti2d(
                        results_, self.CLASSES, pklfile_prefix_,
                        submission_prefix_)
                else:
                    result_files_ = self.bbox2result_kitti(
                        results_, self.CLASSES, pklfile_prefix_,
                        submission_prefix_)
                result_files[name] = result_files_
        else:
            result_files = self.bbox2result_kitti(outputs, self.CLASSES,
                                                  pklfile_prefix,
                                                  submission_prefix)
        return result_files, tmp_dir

    def format_gt(self,
                       gt,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format suscape gt to kitti format.      
        """

      
        gt_annos = []
        print('\nConverting ground truth to KITTI format')

        rt_mat = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

        for  frame in gt:
            annos = {}
            # info = self.data_infos[idx]
            #sample_idx = info['image']['image_idx']
            #image_shape = info['image']['image_shape'][:2]
            boxes = np.array(frame['gt_boxes'])
            pcd_range = np.array(self.pcd_limit_range).reshape((2,3))
            box_range_filter = (boxes[:, :3] >= pcd_range[0,:]) & (boxes[:, :3] <= pcd_range[1,:])
            filter_ind = box_range_filter.all(axis=1)
            boxes = boxes[filter_ind]
            if len(boxes) > 0:
                # box_2d_preds = box_dict['bbox']
               
                annos['name'] = np.array(frame['gt_names'])[filter_ind]
                annos['truncated'] = np.zeros(len(boxes))
                annos['occluded'] = np.zeros(len(boxes))
                annos['alpha'] = np.ones(len(boxes))* (-10) #-np.arctan2(-box[:,1], box[:,0]) + box[:,6]
                annos['bbox'] = np.zeros((len(boxes), 4))
                annos['dimensions'] = np.array(list(map(lambda x: [x[0], x[2], x[1]], boxes[:, 3:6])))#box[:, 3:6]
                annos['location'] = np.array(list(map(lambda x: [-x[1],-(x[2]-x[5]/2), x[0]], boxes[:, :6])))
                annos['rotation_y'] = -boxes[:, 6] - np.pi/2
                annos['score'] = np.ones(len(boxes))
                annos['difficulty'] = np.zeros(len(boxes), np.int32)
            else:
                annos = {
                    'name': np.array([]),
                    'truncated': np.zeros([0]),
                    'occluded': np.zeros([0]),
                    'alpha': np.zeros([0]),
                    'bbox': np.zeros([0,4]),
                    'dimensions': np.zeros([0,3]),
                    'location': np.zeros([0,3]),
                    'rotation_y': np.zeros([0]),
                    'score': np.zeros([0]),
                    'difficulty': np.zeros([0]),
                }
            # annos[-1]['sample_idx'] = np.array(
            #     [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            gt_annos.append(annos)


        return gt_annos
        
    

    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert results to kitti format for evaluation and test submission.

        Args:
            net_outputs (List[np.ndarray]): list of array storing the
                bbox and score
            class_nanes (List[String]): A list of class names
            pklfile_prefix (str): The prefix of pkl file.
            submission_prefix (str): The prefix of submission file.

        Returns:
            List[dict]: A list of dict have the kitti 3d format
        """
        assert len(net_outputs) == len(self.data_infos['data_list']), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos['data_list'][idx]
            #sample_idx = info['image']['image_idx']
            #image_shape = info['image']['image_shape'][:2]

            box_dict = self.convert_valid_bboxes(pred_dicts["boxes_3d"],pred_dicts["scores_3d"], pred_dicts["labels_3d"])
            if len(box_dict['scores']) > 0:
                # box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                anno = {
                    'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []
                }

                for box, box_lidar, score, label in zip(
                        box_preds, box_preds_lidar, scores,
                        label_preds):
                    #bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    #bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(
                        -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    anno['bbox'].append(np.zeros(4))
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

                if submission_prefix is not None:
                    curr_file = f'{submission_prefix}/{sample_idx:07d}.txt'
                    with open(curr_file, 'w') as f:
                        bbox = anno['bbox']
                        loc = anno['location']
                        dims = anno['dimensions']  # lhw -> hwl

                        for idx in range(len(bbox)):
                            print(
                                '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                                '{:.4f} {:.4f} {:.4f} '
                                '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.
                                format(anno['name'][idx], anno['alpha'][idx],
                                       bbox[idx][0], bbox[idx][1],
                                       bbox[idx][2], bbox[idx][3],
                                       dims[idx][1], dims[idx][2],
                                       dims[idx][0], loc[idx][0], loc[idx][1],
                                       loc[idx][2], anno['rotation_y'][idx],
                                       anno['score'][idx]),
                                file=f)
            else:
                annos.append({
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                })
            # annos[-1]['sample_idx'] = np.array(
            #     [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def convert_valid_bboxes(self, boxes, scores, labels):
        """Convert the boxes into valid format.

        Args:
            - boxes (:obj:``LiDARInstance3DBoxes``): 3D bounding boxes.
            - scores (np.ndarray): Scores of predicted boxes.
            - labels (np.ndarray): Class labels of predicted boxes.
        Returns:
            dict: Valid boxes after conversion.

                - bbox (np.ndarray): 2D bounding boxes (in camera 0).
                - box3d_camera (np.ndarray): 3D boxes in camera coordinates.
                - box3d_lidar (np.ndarray): 3D boxes in lidar coordinates.
                - scores (np.ndarray): Scores of predicted boxes.
                - label_preds (np.ndarray): Class labels of predicted boxes.
                - sample_idx (np.ndarray): Sample index.
        """
        # TODO: refactor this function
  
        # sample_idx = info['image']['image_idx']
        boxes.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(boxes) == 0:
            return dict(
                # bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                # sample_idx=sample_idx
                )

        # rect = info['calib']['R0_rect'].astype(np.float32)
        # Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        # P0 = info['calib']['P0'].astype(np.float32)
        # P0 = box_preds.tensor.new_tensor(P0)

        box_preds_camera = boxes.convert_to(Box3DMode.CAM)

        # box_corners = box_preds_camera.corners
        # box_corners_in_image = points_cam2img(box_corners, P0)
        # # box_corners_in_image: [N, 8, 2]
        # minxy = torch.min(box_corners_in_image, dim=1)[0]
        # maxxy = torch.max(box_corners_in_image, dim=1)[0]
        # box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds
        limit_range = boxes.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = ((boxes.center > limit_range[:3]) &
                          (boxes.center < limit_range[3:]))
        valid_inds = valid_pcd_inds.all(-1)

        if valid_inds.sum() > 0:
            return dict(
                # bbox=box_2d_preds[valid_inds, :].numpy(),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                box3d_lidar=boxes[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                # sample_idx=sample_idx,
            )
        else:
            return dict(
                # bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                # sample_idx=sample_idx,
            )
