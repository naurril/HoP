
import os
import json
import re
import numpy as np




class BinPcdReader:
    """ Read binary PCD files.
    """
    def __init__(self, filename):
        self.filename = filename
        self.metadata = None
        self.points = None


        numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
        #numpy_type_to_pcd_type = dict(numpy_pcd_type_mappings)
        self.pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)


        self._read()

    def _read(self):
        with open(self.filename, 'rb') as f:
            header = []
            while True:
                ln = f.readline().strip()
                header.append(ln)
                if ln.startswith('DATA'.encode()):
                    break

            metadata = self.parse_header(header)
            dtype = self._build_dtype(metadata)
            self.pc_data = self.parse_binary_pc_data(f, dtype, metadata)

  
    def parse_binary_pc_data(self, f, dtype, metadata):
        # print("the dtype.itemsize is: ",dtype.itemsize)
        # print("the dtype['x'] is: ",dtype['x'])
        rowstep = metadata['points']*dtype.itemsize
        # for some reason pcl adds empty space at the end of files
        buf = f.read(rowstep)
        return np.fromstring(buf, dtype=dtype)
    
    def parse_header(self, lines):
        """ Parse header of PCD files.
        """
        metadata = {}
        for ln in lines:
            if ln.startswith('#'.encode()) or len(ln) < 2:
                continue
            match = re.match('(\w+)\s+([\w\s\.]+)', ln.decode())
            if not match:
                warnings.warn("warning: can't understand line: %s" % ln)
                continue
            key, value = match.group(1).lower(), match.group(2)
            if key == 'version':
                metadata[key] = value
            elif key in ('fields', 'type'):
                metadata[key] = value.split()
            elif key in ('size', 'count'):
                metadata[key] = [int(i) for i in value.split()]
                # metadata[key] = map(int, value.split())
            elif key in ('width', 'height', 'points'):
                metadata[key] = int(value)
            elif key == 'viewpoint':
                metadata[key] = map(float, value.split())
            elif key == 'data':
                metadata[key] = value.strip().lower()
            # TODO apparently count is not required?
        # add some reasonable defaults
        if 'count' not in metadata:
            metadata['count'] = [1]*len(metadata['fields'])
        if 'viewpoint' not in metadata:
            metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        if 'version' not in metadata:
            metadata['version'] = '.7'
        return metadata

    def _metadata_is_consistent(metadata):
        """ Sanity check for metadata. Just some basic checks.
        """
        checks = []
        required = ('version', 'fields', 'size', 'width', 'height', 'points',
                    'viewpoint', 'data')
        for f in required:
            if f not in metadata:
                print('%s required' % f)
        checks.append((lambda m: all([k in m for k in required]),
                    'missing field'))
        # print("te len of the list(m['count']) is: ",list(m['count']))
        checks.append((lambda m: len(m['type']) == len(list(m['count'])) ==
                    len(list(m['fields'])),
                    'length of type, count and fields must be equal'))
        checks.append((lambda m: m['height'] > 0,
                    'height must be greater than 0'))
        checks.append((lambda m: m['width'] > 0,
                    'width must be greater than 0'))
        checks.append((lambda m: m['points'] > 0,
                    'points must be greater than 0'))
        checks.append((lambda m: m['data'].lower() in ('ascii', 'binary',
                    'binary_compressed'),
                    'unknown data type:'
                    'should be ascii/binary/binary_compressed'))
        ok = True
        for check, msg in checks:
            if not check(metadata):
                print('error:', msg)
                ok = False
        return ok

    def _build_dtype(self, metadata):
        """ Build numpy structured array dtype from pcl metadata.

        Note that fields with count > 1 are 'flattened' by creating multiple
        single-count fields.

        *TODO* allow 'proper' multi-count fields.
        """
        fieldnames = []
        typenames = []
        other = 1
        for f, c, t, s in zip(metadata['fields'],
                            metadata['count'],
                            metadata['type'],
                            metadata['size']):
            np_type = self.pcd_type_to_numpy_type[(t, s)]
            if c == 1:
                fieldnames.append(f)
                typenames.append(np_type)
            else:
                fieldnames.extend(['%s_%s_%04d' % (f,str(other), i) for i in range(c)])
                typenames.extend([np_type]*c)
                other = other + 1
        a = list(zip(fieldnames, typenames))
        dtype = np.dtype(list(zip(fieldnames, typenames)))
        return dtype
        
class SuscapeDataset:
    def __init__(self, root_dir, dir_org='by_scene') -> None:
        
        cfg = self.build_dataset_cfgs(root_dir, dir_org)
        self.camera_dir = cfg['camera']
        self.lidar_dir = cfg['lidar']
        self.aux_camera_dir = cfg['aux_camera']
        self.label_dir = cfg['label']
        self.calib_dir = cfg['calib']
        self.desc_dir = cfg['desc']
        self.meta_dir = cfg['meta']
        self.radar_dir = cfg['radar']
        self.aux_lidar_dir = cfg['aux_lidar']
        self.label_fusion_dir = cfg['label_fusion']
        self.lidar_pose_dir = cfg['lidar_pose']

        pass

    def build_dataset_cfgs(self, root, dir_org='by_scene'):
        dataset_cfg={}
        if dir_org == 'by_scene':
            
            for d in ['lidar',  'label', 'camera', 'calib', 'aux_lidar', 'aux_camera', 'radar', 'desc', 'meta','label_fusion', 'lidar_pose']:
                dataset_cfg[d] = root
            dataset_cfg['root'] = root
            
        elif dir_org == 'by_data_folder':
            for d in ['lidar',  'label', 'camera', 'calib', 'aux_lidar', 'aux_camera', 'radar', 'desc', 'meta','label_fusion', 'lidar_pose']:
                dataset_cfg[d] = datacfg['global'][d] if d in datacfg['global'] else (root + '/' + d)
            dataset_cfg['root'] = root
        else:
            print("data cfg error.")
        
        return dataset_cfg

    def get_all_scene_desc(self, scene_pattern):
        
        scenes = self.get_scene_names()
        
        descs = {}

        for n in scenes:
            if re.fullmatch(scene_pattern, n):
                try:
                    descs[n] = self.get_scene_desc(n)
                except:
                    print('failed reading scene:', n)
                    raise
        return descs

    def get_scene_names(self):
        scenes = os.listdir(self.lidar_dir)
        scenes.sort()
        return scenes

    

    def get_all_objs(self, s):
      label_folder = os.path.join(self.label_dir, s, "label")
      if not os.path.isdir(label_folder):
        return []
        
      files = os.listdir(label_folder)

      files = filter(lambda x: x.split(".")[-1]=="json", files)


      def file_2_objs(f):
          if  not os.path.exists(f):
            return []
          with open(f) as fd:
              ann = json.load(fd)
              if 'objs' in ann:
                boxes = ann['objs']
              else:
                boxes = ann
              objs = [x for x in map(lambda b: {"category":b["obj_type"], "id": b["obj_id"]}, boxes)]
              return objs

      boxes = map(lambda f: file_2_objs(os.path.join(self.label_dir, s, "label", f)), files)

      # the following map makes the category-id pairs unique in scene
      all_objs={}
      for x in boxes:
          for o in x:
              
              k = str(o["category"])+"-"+str(o["id"])

              if all_objs.get(k):
                all_objs[k]['count']= all_objs[k]['count']+1
              else:
                all_objs[k]= {
                  "category": o["category"],
                  "id": o["id"],
                  "count": 1
                }

      return [x for x in  all_objs.values()]



    
    def get_calib_lidar2cam(self, scene_info, frame, camera_type, camera_name):
        s = scene_info["scene"]
        static_calib = np.array(scene_info["calib"][camera_type][camera_name]["lidar_to_camera"]).reshape([4,4])

        frame_calib_file = os.path.join(self.calib_dir, s, "calib", camera_type, camera_name, frame+".json")
        if not os.path.exists(frame_calib_file):
            return static_calib
        with open(frame_calib_file) as f:
            local_calib = json.load(f)
            trans = np.array(local_calib['lidar_transform']).reshape((4,4))
            lidar2cam = np.matmul(static_calib, trans)
            return lidar2cam
    
    def get_scene_desc(self, s):
        scene_dir = os.path.join(self.desc_dir, s)
        desc = {}
        if os.path.exists(os.path.join(scene_dir, "desc.json")):
            with open(os.path.join(scene_dir, "desc.json")) as f:
                desc = json.load(f)
        
        return desc
        

    def get_scene_info(self, s):
        scene = {
            "scene": s,
            "frames": []
        }

        frames = os.listdir(os.path.join(self.lidar_dir, s, "lidar"))

        frames.sort()

        scene["lidar_ext"]="pcd"
        for f in frames:
            #if os.path.isfile("./data/"+s+"/lidar/"+f):
            filename, fileext = os.path.splitext(f)
            scene["frames"].append(filename)
            scene["lidar_ext"] = fileext

        # point_transform_matrix=[]

        # if os.path.isfile(os.path.join(scene_dir, "point_transform.txt")):
        #     with open(os.path.join(scene_dir, "point_transform.txt"))  as f:
        #         point_transform_matrix=f.read()
        #         point_transform_matrix = point_transform_matrix.split(",")

        
        if os.path.exists(os.path.join(self.desc_dir, s, "desc.json")):
            with open(os.path.join(self.desc_dir, s, "desc.json")) as f:
                desc = json.load(f)
                scene["desc"] = desc

        # calib will be read when frame is loaded. since each frame may have different calib.
        # read default calib for whole scene.
        calib = {}
        if os.path.exists(os.path.join(self.calib_dir, s, "calib")):
            sensor_types = os.listdir(os.path.join(self.calib_dir, s, 'calib'))        
            for sensor_type in sensor_types:
                calib[sensor_type] = {}
                if os.path.exists(os.path.join(self.calib_dir, s, "calib",sensor_type)):
                    calibs = os.listdir(os.path.join(self.calib_dir, s, "calib", sensor_type))
                    for c in calibs:
                        calib_file = os.path.join(self.calib_dir, s, "calib", sensor_type, c)
                        calib_name, ext = os.path.splitext(c)
                        if os.path.isfile(calib_file) and ext==".json": #ignore directories.
                            #print(calib_file)
                            try:
                                with open(calib_file)  as f:
                                    cal = json.load(f)
                                    calib[sensor_type][calib_name] = cal
                            except: 
                                print('reading calib failed: ', f)
                                assert False, f            


        scene["calib"] = calib


        # camera names
        camera = []
        camera_ext = ""
        cam_path = os.path.join(self.camera_dir, s, "camera")
        if os.path.exists(cam_path):
            cams = os.listdir(cam_path)
            for c in cams:
                cam_file = os.path.join(self.camera_dir, s, "camera", c)
                if os.path.isdir(cam_file):
                    camera.append(c)

                    if camera_ext == "":
                        #detect camera file ext
                        files = os.listdir(cam_file)
                        if len(files)>=2:
                            _,camera_ext = os.path.splitext(files[0])

        camera.sort()
        if camera_ext == "":
            camera_ext = ".jpg"
        scene["camera_ext"] = camera_ext
        scene["camera"] = camera


        aux_camera = []
        aux_camera_ext = ""
        aux_cam_path = os.path.join(self.aux_camera_dir, s, "aux_camera")
        if os.path.exists(aux_cam_path):
            cams = os.listdir(aux_cam_path)
            for c in cams:
                cam_file = os.path.join(aux_cam_path, c)
                if os.path.isdir(cam_file):
                    aux_camera.append(c)

                    if aux_camera_ext == "":
                        #detect camera file ext
                        files = os.listdir(cam_file)
                        if len(files)>=2:
                            _,aux_camera_ext = os.path.splitext(files[0])

        aux_camera.sort()
        if aux_camera_ext == "":
            aux_camera_ext = ".jpg"
        scene["aux_camera_ext"] = aux_camera_ext
        scene["aux_camera"] = aux_camera


        # radar names
        radar = []
        radar_ext = ""
        radar_path = os.path.join(self.radar_dir, s, "radar")
        if os.path.exists(radar_path):
            radars = os.listdir(radar_path)
            for r in radars:
                radar_file = os.path.join(self.radar_dir, s, "radar", r)
                if os.path.isdir(radar_file):
                    radar.append(r)
                    if radar_ext == "":
                        #detect camera file ext
                        files = os.listdir(radar_file)
                        if len(files)>=2:
                            _,radar_ext = os.path.splitext(files[0])

        if radar_ext == "":
            radar_ext = ".pcd"
        scene["radar_ext"] = radar_ext
        scene["radar"] = radar

        # aux lidar names
        aux_lidar = []
        aux_lidar_ext = ""
        aux_lidar_path = os.path.join(self.aux_lidar_dir, s, "aux_lidar")
        if os.path.exists(aux_lidar_path):
            lidars = os.listdir(aux_lidar_path)
            for r in lidars:
                lidar_file = os.path.join(self.aux_lidar_dir, s, "aux_lidar", r)
                if os.path.isdir(lidar_file):
                    aux_lidar.append(r)
                    if radar_ext == "":
                        #detect camera file ext
                        files = os.listdir(radar_file)
                        if len(files)>=2:
                            _,aux_lidar_ext = os.path.splitext(files[0])

        if aux_lidar_ext == "":
            aux_lidar_ext = ".pcd"
        scene["aux_lidar_ext"] = aux_lidar_ext
        scene["aux_lidar"] = aux_lidar


        scene["boxtype"] = "psr"

        # lidar_pose
        lidar_pose= {}
        lidar_pose_path = os.path.join(self.lidar_pose_dir, s, "lidar_pose")
        if os.path.exists(lidar_pose_path):
            poses = os.listdir(lidar_pose_path)
            for p in poses:
                p_file = os.path.join(lidar_pose_path, p)
                with open(p_file)  as f:
                        pose = json.load(f)
                        lidar_pose[os.path.splitext(p)[0]] = pose
        
        scene['lidar_pose'] = lidar_pose
        return scene


    def read_label(self, scene, frame):
        "read 3d boxes"
        if not os.path.exists(os.path.join(self.label_dir, scene, 'label')):
            print('label path does not exist', self.label_dir, scene, 'label')
            return {'objs': []}
        
        filename = os.path.join(self.label_dir, scene, "label", frame+".json")   # backward compatible
        
        if os.path.exists(filename):
            if (os.path.isfile(filename)):
                with open(filename,"r") as f:
                    ann=json.load(f)
                    return ann
        else:
            print('label file does not exist', filename)
        return {'objs': []}


    def read_image_annotations(self, scene, frame, camera_type, camera_name):
        filename = os.path.join(self.label_fusion_dir, scene, "label_fusion", camera_type, camera_name, frame+".json")   # backward compatible
        if os.path.exists(filename):
            if (os.path.isfile(filename)):
                with open(filename,"r") as f:
                    ann=json.load(f)
                    #print(ann)          
                    return ann
        return {'objs': []}


    def read_all_image_annotations(self, scene, frame, cameras, aux_cameras):
        ann = {
            "camera": {},
            "aux_camera": {}
        }
        for c in cameras.split(','):
            filename = os.path.join(self.label_fusion_dir, scene, "label_fusion", 'camera', c, frame+".json")   # backward compatible
            if os.path.exists(filename):
                if (os.path.isfile(filename)):
                    with open(filename,"r") as f:
                        ann['camera'][c] = json.load(f)


        for c in aux_cameras.split(','):
            filename = os.path.join(self.label_fusion_dir, scene, "label_fusion", 'aux_camera', c, frame+".json")   # backward compatible
            if os.path.exists(filename):
                if (os.path.isfile(filename)):
                    with open(filename,"r") as f:
                        ann['aux_camera'][c] = json.load(f)

        return ann

    def read_lidar_pose(self, scene, frame):
        filename = os.path.join(self.lidar_pose_dir, scene, "lidar_pose", frame+".json")
        if (os.path.isfile(filename)):
            with open(filename,"r") as f:
                p=json.load(f)
                return p
        else:
            return None
        
    def read_calib(self, scene, frame):
        'read static calibration, all extrinsics are sensor to lidar_top'
        calib = {}

        if not os.path.exists(os.path.join(self.calib_dir, scene, "calib")):
            return calib

        calib_folder = os.path.join(self.calib_dir, scene, "calib")
        sensor_types = os.listdir(calib_folder)
        
        for sensor_type in sensor_types:
            this_type_calib = {}
            sensors = os.listdir(os.path.join(calib_folder, sensor_type))
            for sensor in sensors:
                sensor_file = os.path.join(calib_folder, sensor_type, sensor, frame+".json")
                if os.path.exists(sensor_file) and os.path.isfile(sensor_file):
                    with open(sensor_file, "r") as f:
                        p=json.load(f)
                        this_type_calib[sensor] = p
            if this_type_calib:
                calib[sensor_type] = this_type_calib

        return calib


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='start web server for SUSTech POINTS')        
    parser.add_argument('data', type=str, help='data folder')
    args = parser.parse_args()

    dataset = SuscapeDataset(args.data)
    print(len(dataset.get_scene_names()), 'scenes')
    print(dataset.get_scene_info("scene-000000"))