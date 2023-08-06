# Copyright (c) OpenMMLab. All rights reserved.
import os
from logging import warning
from os import path as osp
import mmengine
import numpy as np
import json
from .suscape_dataset import SuscapeDataset, BinPcdReader
from pyquaternion import Quaternion

## pcd parser

import re
import warnings




####################################################################3


def _read_list_from_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]



METAINFO = {
    'classes': ['Car', 'Pedestrian', 'ScooterRider', 'Truck', 'Scooter',
                'Bicycle', 'Van', 'Bus', 'BicycleRider', 
                #'BicycleGroup', 
                'Trimotorcycle', #'RoadWorker', 
                
                # 'LongVehicle', 'Cone', 
                # 'TrafficBarrier', 'ConcreteTruck', 'Child', 'BabyCart', 
                # 'RoadBarrel', #'FireHydrant', 
                # #'MotorcyleRider', 
                # 'Crane', 
                # 'ForkLift', 'Bulldozer', 'Excavator', 
                #'Motorcycle'
                ],     
    'classMap': {
        'Motorcycle': 'Scooter',
        'MotorcyleRider': 'ScooterRider',
        'RoadWorker': 'Pedestrian',
        'LongVehicle': 'Truck',
        'ConcreteTruck': 'Truck',
        'Child': 'Pedestrian',
        'Crane': 'Truck',
        'ForkLift': 'Truck',
        'Bulldozer': 'Truck', 
        'Excavator': 'Truck',
    }
}

def mapclassname(n):
    return METAINFO['classMap'][n] if n in METAINFO['classMap'] else n

def class2id(n):
    n = mapclassname(n)
    return METAINFO['classes'].index(n)

def create_suscape_infos(root_path,
                      info_prefix,
                      out_path,
                      version='v1.0-train',
                      max_sweeps=1):
    """Create info file of suscape dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-train'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 1.
    """

    available_vers = ['v1.0-trainval', 'v1.0-test']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = _read_list_from_file('data/suscape/train.txt')
        val_scenes = _read_list_from_file('data/suscape/val.txt')
    elif version == 'v1.0-test':
        train_scenes = _read_list_from_file('data/suscape/test.txt')
        val_scenes = []
    else:
        raise ValueError('unknown')

    suscape = SuscapeDataset(root_path)
    # filter existing scenes.
    available_scenes = suscape.get_scene_names()
    

    available_scene_names = available_scenes
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))

    test = 'test' in version
    if test:
        print(f'test scene: {len(train_scenes)}')
    else:
        print(f'train scene: {len(train_scenes)}, \
                val scene: {len(val_scenes)}')
        
    train_infos, val_infos = _fill_trainval_infos(
        suscape, train_scenes, val_scenes, out_path, test, max_sweeps=max_sweeps)

    metainfo = dict(version=version, dataset='suscape')
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    metainfo['version'] = version
    metainfo['info_version'] = '1.1'

    if test:
        print(f'test sample: {len(train_infos)}')
        data = dict(data_list=train_infos, metainfo=metainfo)
        info_name = f'{info_prefix}_infos_test'
        info_path = osp.join(out_path, f'{info_name}.pkl')
        mmengine.dump(data, info_path)
    else:
        print(f'train sample: {len(train_infos)}, \
                val sample: {len(val_infos)}')
        data = dict(data_list=train_infos, metainfo=metainfo)
        train_info_name = f'{info_prefix}_infos_train'
        info_path = osp.join(out_path, f'{train_info_name}.pkl')
        mmengine.dump(data, info_path)
        data = dict(data_list=val_infos, metainfo=metainfo)
        val_info_name = f'{info_prefix}_infos_val'
        info_val_path = osp.join(out_path, f'{val_info_name}.pkl')
        mmengine.dump(data, info_val_path)

def _read_image_info(root_path, scene, frame, camera_type, camera):
    with open(os.path.join(root_path, scene, 'calib', camera_type, camera+'.json')) as f:
        calib = json.load(f)

        return {
            'lidar2cam': np.reshape(np.array(calib['extrinsic']),(4,4)),
            'cam2img': np.reshape(np.array(calib['intrinsic']), (3,3)),     
            'height': 1536,
            'width': 2048,      
        }


def _read_scene(suscape, out_path, scene_name, test, overwrite_lidar_file=False):
    
    scene = suscape.get_scene_info(scene_name)

    infos = []
    for frame in scene['frames']:
        
        # write lidar bin files
        bin_lidar_path = osp.join(out_path, "lidar.bin", scene_name, "lidar")
        if os.path.exists(bin_lidar_path) == False:
            os.makedirs(bin_lidar_path)

        bin_lidar_file = osp.join(bin_lidar_path, frame+".bin")

        if not os.path.exists(bin_lidar_file) or overwrite_lidar_file :
            pcd_reader = BinPcdReader(osp.join(suscape.lidar_dir, scene_name, 'lidar', frame+scene['lidar_ext']))
            
            bin_data = np.stack([pcd_reader.pc_data[x] for x in ['x', 'y', 'z', 'intensity']], axis=-1)
            bin_data = bin_data[(bin_data[:,0]!=0) & (bin_data[:,1]!=0) & (bin_data[:,2]!=0)]
            bin_data.tofile(bin_lidar_file)

        ego2global = np.array(scene['lidar_pose'][frame]['lidarPose']).reshape(4,4)
        info = {

            'lidar_path': os.path.join(out_path, "lidar.bin", scene_name, "lidar", frame+".bin"),
            'sweeps': [],
            'cams': dict(),
            'lidar2ego_translation': [0,0,0],
            'lidar2ego_rotation': [1,0,0,0],
            'ego2global_translation': ego2global[:3,3].tolist(),
            'ego2global_rotation': Quaternion(matrix=ego2global).elements,
            #'timestamp': sample['timestamp'],
        }

        # info['images']['camera/front'] = _read_image_info(root_path, scene, frame, 'camera', 'front')

       
        for cam in scene['camera']:
            lidar2cam = suscape.get_calib_lidar2cam(scene, frame, 'camera', cam)
            cam2lidar = np.linalg.inv(lidar2cam)
            info['cams'][cam] = {
                'data_path': os.path.join(suscape.camera_dir, scene_name, 'camera', cam, frame+scene['camera_ext']),
                'type': cam,
                'sensor2ego_translation': cam2lidar[:3,3].tolist(),
                'sensor2ego_rotation': Quaternion(matrix=cam2lidar).elements,
                'ego2global_translation': ego2global[:3,3].tolist(),
                'ego2global_rotation': Quaternion(matrix=ego2global).elements,
                'sensor2lidar_translation': cam2lidar[:3,3],
                'sensor2lidar_rotation': cam2lidar[:3,:3],
                'cam_intrinsic': np.array(scene['calib']['camera'][cam]['intrinsic']).reshape(3,3),
            }

        # no sweeps for now

        # labels
        if not test:
            objs = suscape.read_label(scene_name, frame)['objs']

            objs = list(filter(lambda o: mapclassname(o['obj_type']) in METAINFO['classes'], objs))

            if len(objs) == 0:
                print("no objs: ", osp.join(scene_name, 'label', frame+".json"))
                continue

            info['gt_boxes'] = np.array([[o['psr']['position']['x'],
                                            o['psr']['position']['y'],
                                            o['psr']['position']['z'],
                                            o['psr']['scale']['x'],
                                            o['psr']['scale']['y'],
                                            o['psr']['scale']['z'],
                                            o['psr']['rotation']['z'],
                                            ] for o in objs])
            info['gt_names'] = np.array([mapclassname(o['obj_type']) for o in objs])
            info['gt_velocity'] = np.array([0 for o in objs])
            info['num_lidar_pts'] = np.array([100 for o in objs])
            info['num_radar_pts'] = np.array([0 for o in objs])
            info['valid_flag'] = np.array([True for o in objs])

            info['ann_infos']=[
                np.concatenate([info['gt_boxes'], np.zeros([info['gt_boxes'].shape[0], 2])], axis=1),
                [METAINFO['classes'].index(n) for n in info['gt_names']],
            ]
            info['scene_token'] = scene_name # useful to find adjecent frames
            infos.append(info)
    return infos

def _fill_trainval_infos(suscape,
                         train_scenes,
                         val_scenes,
                         out_path,
                         test=False,
                         max_sweeps=1):
    """Generate the train/val infos from the raw data.

    Args:
        root_path: suscape dataset root path
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 1.

    Returns:
        tuple[list[dict]]: Information of training set and
            validation set that will be saved to the info file.
    """
    train_infos = []
    val_infos = []

    for scene in train_scenes:
        print(scene)
        infos = _read_scene(suscape, out_path, scene, test)
        train_infos.extend(infos)
    for scene in val_scenes:
        print(scene)
        infos = _read_scene(suscape, out_path, scene, test)
        val_infos.extend(infos)
    return train_infos, val_infos


# suscape info example
# info
# {'lidar_path': './data/nuscenes/samp...77.pcd.bin', 'token': 'e93e98b63d3b40209056...29dc53ceee', 'sweeps': [], 'cams': {'CAM_FRONT': {...}, 'CAM_FRONT_RIGHT': {...}, 'CAM_FRONT_LEFT': {...}, 'CAM_BACK': {...}, 'CAM_BACK_LEFT': {...}, 'CAM_BACK_RIGHT': {...}}, 'lidar2ego_translation': [0.943713, 0.0, 1.84023], 'lidar2ego_rotation': [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817], 'ego2global_translation': [1010.1328353833223, 610.8111652918716, 0.0], 'ego2global_rotation': [-0.7495886280607293, -0.0077695335695504636, 0.00829759813869316, -0.6618063711504101], 'timestamp': 1531883530449377, 'gt_boxes': array([[-1.61843454e...048e-01]]), 'gt_names': array(['traffic_cone...pe='<U12'), 'gt_velocity': array([[-7.37269312e...000e+00]]), 'num_lidar_pts': array([  2,   3, 171...  4,  42]), 'num_radar_pts': array([0, 0, 7, 2, 0... 0, 0, 6]), ...}
# special variables:
# function variables:
# 'lidar_path': './data/nuscenes/samples/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530449377.pcd.bin'
# 'token': 'e93e98b63d3b40209056d129dc53ceee'
# 'sweeps': []
# 'cams': {'CAM_FRONT': {'data_path': './data/nuscenes/samp...412470.jpg', 'type': 'CAM_FRONT', 'sample_data_token': '020d7b4f858147558106...04f7f31bef', 'sensor2ego_translation': [...], 'sensor2ego_rotation': [...], 'ego2global_translation': [...], 'ego2global_rotation': [...], 'timestamp': 1531883530412470, 'sensor2lidar_rotation': array([[ 0.99995012,...1906509]]), ...}, 'CAM_FRONT_RIGHT': {'data_path': './data/nuscenes/samp...420339.jpg', 'type': 'CAM_FRONT_RIGHT', 'sample_data_token': '16d39ff22a8545b0a4ee...36a0fe1c20', 'sensor2ego_translation': [...], 'sensor2ego_rotation': [...], 'ego2global_translation': [...], 'ego2global_rotation': [...], 'timestamp': 1531883530420339, 'sensor2lidar_rotation': array([[ 0.5447327 ,...0506993]]), ...}, 'CAM_FRONT_LEFT': {'data_path': './data/nuscenes/samp...404844.jpg', 'type': 'CAM_FRONT_LEFT', 'sample_data_token': '24332e9c554a406f8804...f17771b608', 'sensor2ego_translation': [...], 'sensor2ego_rotation': [...], 'ego2global_translation': [...], 'ego2global_rotation': [...], 'timestamp': 1531883530404844, 'sensor2lidar_rotation': array([[ 0.58312896,...1142902]]), ...}, 'CAM_BACK': {'data_path': './data/nuscenes/samp...437525.jpg', 'type': 'CAM_BACK', 'sample_data_token': 'aab35aeccbda42de82b2...5c278a0d48', 'sensor2ego_translation': [...], 'sensor2ego_rotation': [...], 'ego2global_translation': [...], 'ego2global_rotation': [...], 'timestamp': 1531883530437525, 'sensor2lidar_rotation': array([[-0.99991364,...0763594]]), ...}, 'CAM_BACK_LEFT': {'data_path': './data/nuscenes/samp...447423.jpg', 'type': 'CAM_BACK_LEFT', 'sample_data_token': '86e6806d626b4711a6d0...015b090116', 'sensor2ego_translation': [...], 'sensor2ego_rotation': [...], 'ego2global_translation': [...], 'ego2global_rotation': [...], 'timestamp': 1531883530447423, 'sensor2lidar_rotation': array([[-0.31651335,...2949614]]), ...}, 'CAM_BACK_RIGHT': {'data_path': './data/nuscenes/samp...427893.jpg', 'type': 'CAM_BACK_RIGHT', 'sample_data_token': 'ec7096278e484c9ebe68...a2ad5682e9', 'sensor2ego_translation': [...], 'sensor2ego_rotation': [...], 'ego2global_translation': [...], 'ego2global_rotation': [...], 'timestamp': 1531883530427893, 'sensor2lidar_rotation': array([[-0.36268682,...1896112]]), ...}}
# 'lidar2ego_translation': [0.943713, 0.0, 1.84023]
# 'lidar2ego_rotation': [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817]
# 'ego2global_translation': [1010.1328353833223, 610.8111652918716, 0.0]
# 'ego2global_rotation': [-0.7495886280607293, -0.0077695335695504636, 0.00829759813869316, -0.6618063711504101]
# 'timestamp': 1531883530449377
# 'gt_boxes': array([[-1.61843454e+01, -1.17404151e+00, -1.24046699e+00,
#          2.91000000e-01,  3.00000000e-01,  7.34000000e-01,
#         -2.93534163e+00],
#        [-1.54493912e+01, -4.28768163e+00, -1.30136452e+00,
#          3.38000000e-01,  3.15000000e-01,  7.12000000e-01,
#         -2.83072660e+00],
#        [-1.02275670e+01,  1.94608211e+01,  3.74364245e-02,
#          7.51600000e+00,  2.31200000e+00,  3.09300000e+00,
#         -6.06636077e-01],
#        [ 9.21442005e+00, -5.57960735e+00, -1.07856950e+00,
#          4.25000000e+00,  1.63800000e+00,  1.44000000e+00,
#          3.58496093e-01],
#        [-1.57271212e+01, -8.16090985e-01, -6.97936424e-01,
#          5.63000000e-01,  7.39000000e-01,  1.71100000e+00,
#         -2.93534163e+00],
#        [ 3.84646471e-01, -1.32284491e+01, -1.21462740e+00,
#          4.47800000e+00,  1.87100000e+00,  1.45600000e+00,
#          1.09320989e+00],
#        [-4.75276596e+01,  3.51366615e+01,  6.94957388e-01,
#          6.37200000e+00,  2.87700000e+00,  2.97800000e+00,
#          2.58914905e+00],
#        [-1.61056541e+01, -7.16475402e-02, -6.86282715e-01,
#          5.44000000e-01,  6.65000000e-01,  1.73900000e+00,
#         -2.93534163e+00],
#        [-1.59411481e+01, -2.44787704e+00, -1.28580015e+00,
#          3.09000000e-01,  3.38000000e-01,  7.12000000e-01,
#         -2.92877187e+00],
#        [-1.93828613e+01,  2.55393813e+01,  3.19190807e-02,
#          6.22700000e+00,  2.15600000e+00,  2.60100000e+00,
#         -5.10800048e-01]])
# 'gt_names': array(['traffic_cone', 'traffic_cone', 'truck', 'car', 'pedestrian',
#        'car', 'truck', 'pedestrian', 'traffic_cone', 'truck'],
#       dtype='<U12')
# 'gt_velocity': array([[-7.37269312e-03,  2.72807041e-02],
#        [-2.31282614e-02,  7.36653392e-02],
#        [ 5.79080632e-02,  6.95126389e-02],
#        [ 5.52849014e+00,  1.19551431e+00],
#        [-2.62910013e-02,  9.94492146e-02],
#        [ 2.43858125e+00,  4.21280600e+00],
#        [ 4.97684757e-03,  8.66331948e-03],
#        [ 9.17358141e-02,  1.21979102e-01],
#        [-4.90278498e-03,  2.35572272e-02],
#        [ 0.00000000e+00,  0.00000000e+00]])
# 'num_lidar_pts': array([  2,   3, 171, 150,   7, 151,   9,  10,   4,  42])
# 'num_radar_pts': array([0, 0, 7, 2, 0, 3, 3, 0, 0, 6])
# 'valid_flag': array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
#         True])
# len(): 15



# 'CAM_FRONT': {'data_path': './data/nuscenes/samp...412470.jpg', 'type': 'CAM_FRONT', 'sample_data_token': '020d7b4f858147558106...04f7f31bef', 'sensor2ego_translation': [1.70079118954, 0.0159456324149, 1.51095763913], 'sensor2ego_rotation': [0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755], 'ego2global_translation': [1010.1102882349232, 610.6567106479714, 0.0], 'ego2global_rotation': [-0.7530285141171715, -0.007718682910458633, 0.00863090844122062, -0.6578859979358822], 'timestamp': 1531883530412470, 'sensor2lidar_rotation': array([[ 0.99995012,...1906509]]), 'sensor2lidar_translation': array([ 0.00072265, ...31034774]), 'cam_intrinsic': array([[1.26641720e+...000e+00]])}

