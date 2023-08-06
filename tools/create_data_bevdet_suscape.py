# Copyright (c) OpenMMLab. All rights reserved.
import pickle
import argparse
import numpy as np
from pyquaternion import Quaternion
import pdb

from tools.data_converter import suscape_converter as suscape_converter


classes = ['Car', 'Pedestrian', 'ScooterRider', 'Truck', 'Scooter',
                'Bicycle', 'Van', 'Bus', 'BicycleRider', #'BicycleGroup', 
                'Trimotorcycle', #'RoadWorker', 
                
                # 'LongVehicle', 'Cone', 
                # 'TrafficBarrier', 'ConcreteTruck', 'Child', 'BabyCart', 
                # 'RoadBarrel', #'FireHydrant', 
                # #'MotorcyleRider', 
                # 'Crane', 
                # 'ForkLift', 'Bulldozer', 'Excavator', 
                #'Motorcycle'
                ]

def get_gt(info):
    """Generate gt labels from info.
    Args:
        info(dict): Infos needed to generate gt labels.
    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """

    gt_boxes = list()
    gt_labels = list()
    for ann_info in info['ann_infos']:
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels


def suscape_data_prep(root_path, info_prefix, output_path, version, max_sweeps=10):
    """Prepare data related to nuScenes dataset.
    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.
    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    suscape_converter.create_suscape_infos(
        root_path, info_prefix, output_path, version=version, max_sweeps=max_sweeps)


def add_ann_adj_info(extra_tag, 
                    version='v1.0-trainval', 
                    dataroot='./data/nuscenes'):
    nuscenes = NuScenes(version, dataroot)
    if version == 'v1.0-trainval':
        for set in ['train', 'val']:
            dataset = pickle.load(
                open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
            for id in range(len(dataset['infos'])):
                if id % 10 == 0:
                    print('%d/%d' % (id, len(dataset['infos'])))
                info = dataset['infos'][id]
                # get sweep adjacent frame info
                sample = nuscenes.get('sample', info['token'])

                ann_infos = list()
                for ann in sample['anns']:
                    ann_info = nuscenes.get('sample_annotation', ann)
                    velocity = nuscenes.box_velocity(ann_info['token'])
                    if np.any(np.isnan(velocity)):
                        velocity = np.zeros(3)
                    ann_info['velocity'] = velocity
                    ann_infos.append(ann_info)
                dataset['infos'][id]['ann_infos'] = ann_infos
                dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
                dataset['infos'][id]['scene_token'] = sample['scene_token']
            with open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set),
                    'wb') as fid:
                pickle.dump(dataset, fid)
    elif version == 'v1.0-test':
        set = 'test'
        dataset = pickle.load(
            open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            # get sweep adjacent frame info
            sample = nuscenes.get('sample', info['token'])

            dataset['infos'][id]['scene_token'] = sample['scene_token']
        with open('./data/nuscenes/%s_infos_%s.pkl' % (extra_tag, set),
                'wb') as fid:
            pickle.dump(dataset, fid)
    else:
        raise NotImplementedError(f'{version} not supported')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument('--split', default='trainval', help='split of the dataset')

    args = parser.parse_args()

    dataset = 'suscape'
    version = 'v1.0'
    assert args.split in ['trainval', 'test']
    train_version = f'{version}-{args.split}'
    root_path = './data/suscape/suscape_scenes'
    extra_tag = 'bevdetv2-suscape'
    suscape_data_prep(
        root_path=root_path,
        info_prefix=extra_tag,
        output_path='./data/suscape',
        version=train_version,
        max_sweeps=0)
