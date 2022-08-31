# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp
import json

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, halve_image=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.halve_image = halve_image

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])
            if type(flow) is tuple:
                flow, valid = flow

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if self.halve_image:
            half_width = int(flow.shape[1] / 2)
            if self.extra_info[index]['camera'] == 'GX301187':
                flow = flow[:, :half_width, :]
                img1 = img1[:, :half_width, :]
                img2 = img2[:, :half_width, :]
            else:
                flow = flow[:, half_width:, :]
                img1 = img1[:, half_width:, :]
                img2 = img2[:, half_width:, :]

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()#, self.extra_info[index]

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


class AWI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/AWI'):
        super(AWI, self).__init__(aug_params, sparse=False)

        # Constants
        TARGETS_FILE_ = 'targets_features.json'
        CAMERAS_ = ['GX300643', 'GX301187']
        ANNOTATION_DIR_ = 'json'
        IMAGE_PAIR_DIR_ = 'before_skirt'

        if split == 'test':
            self.is_test = True
        else:
            raise NotImplementedError

        # Load the targets
        targets_file_name = os.path.join(root, TARGETS_FILE_)
        with open(targets_file_name) as json_file:
            fleece_dirs = json.load(json_file)

        # Store the image annotation file names in a list
        fleece_dirs = fleece_dirs[split]

        # Get all images within the named directories
        # Data assumed to be organised as:
        # subdir
        # | - camera1
        #     | - json
        #         | - image1.json
        #         | - ...
        #         | - imageN.json
        #         | - before_skirt
        #             | - image1.png
        #             | - ...
        #             | - imageN.png
        #         | - after_skirt
        #             | - image1.png
        #             | - ...
        #             | - imageN.png
        # | - ...
        # | - cameraN
        #     | ...
        for subdir in fleece_dirs:
            for c in CAMERAS_:
                path_to_annotations = os.path.join(root, subdir, c, ANNOTATION_DIR_)
                # Get all files in the directory
                for i, f in enumerate(os.listdir(path_to_annotations)):
                    anno_file = os.path.join(path_to_annotations, f)
                    if os.path.isfile(anno_file) and f.endswith('.json'):
                        with open(anno_file) as json_file:
                            anno_data = json.load(json_file)
                        image_1 = os.path.join(path_to_annotations, IMAGE_PAIR_DIR_, f.replace('.json', '.png'))
                        image_2 = os.path.join(path_to_annotations, anno_data['correspondence'])

                        # self.image_list += [[image_1, image_2]]
                        self.image_list += [[image_2, image_1]]  # Reversing this because we want flow from after to before skirted
                        self.extra_info += [(subdir, c, f.replace('.json', ''))]  # scene, camera and frame_id


class AWI2(FlowDataset):
    def __init__(self, aug_params=None, halve_image=False, split='training', root='datasets/AWI'):
        super(AWI2, self).__init__(aug_params, sparse=False, halve_image=halve_image)

        # Constants
        CAMERAS_ = ['GX300643', 'GX301187']
        ANNOTATION_DIR_ = 'json'
        IMAGE_PAIR_DIR_ = 'before_skirt'
        VALIDATION_SPLIT_ = ['fleece55_2021-12-16-10-01-43', 'fleece5_2021-12-15-10-59-41',
                             'fleece65_2021-12-16-10-51-22']
        TEST_SPLIT_ = ['fleece15_2021-12-15-11-55-44', 'fleece21_2021-12-15-13-38-58', 'fleece25_2021-12-15-14-07-57',
                       'fleece43_2021-12-16-08-56-20', 'fleece50_2021-12-16-09-31-16', 'fleece56_2021-12-16-10-06-10',
                       'fleece59_2021-12-16-10-20-39', 'fleece70_2021-12-16-11-20-18']

        if split == 'test':
            self.is_test = True

        flow_dir = root.replace('AWI_Dataset', 'AWI_Dataset_Flow_Annotation')

        if split == 'validation':
            fleece_dirs = VALIDATION_SPLIT_
        elif split == 'test':
            fleece_dirs = TEST_SPLIT_
        else:
            reject_list = VALIDATION_SPLIT_ + TEST_SPLIT_
            fleece_dirs = sorted([f for f in os.listdir(flow_dir)
                                  if f.startswith('fleece') and f not in reject_list])

        for subdir in fleece_dirs:
            for c in CAMERAS_:
                path_to_annotations = os.path.join(root, subdir, c, ANNOTATION_DIR_)
                path_to_flows = os.path.join(flow_dir, subdir, c)
                # Get all files in the directory
                for i, f in enumerate(os.listdir(path_to_flows)):
                    flow_file = os.path.join(path_to_flows, f)
                    # flow_gt = sio.loadmat(flow_file)['matrix']
                    anno_file = os.path.join(path_to_annotations, f.replace('.mat', '.json'))
                    if os.path.isfile(anno_file):
                        with open(anno_file) as json_file:
                            anno_data = json.load(json_file)
                        image_1 = os.path.join(path_to_annotations, IMAGE_PAIR_DIR_, f.replace('.json', '.png'))
                        image_2 = os.path.join(path_to_annotations, anno_data['correspondence'])

                        self.image_list += [[image_2, image_1]]  # Reversing this because we want flow from after to before skirted
                        self.flow_list += [flow_file]
                        self.extra_info += [{'scene': subdir, 'camera': c, 'frame': f.replace('.json', '')}]  # scene, camera and frame_id


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding training set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        #things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean', root='/home/tpatten/Data/Opticalflow/Sintel')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final', root='/home/tpatten/Data/Opticalflow/Sintel')

        TRAIN_DS == 'C+T+K/S'

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final# + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    elif args.stage == 'awi':
        input_image_width = 2464
        aug_params = None  # {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        if args.image_size[1] != input_image_width:
            train_dataset = AWI2(aug_params, split='training', halve_image=True)
        else:
            train_dataset = AWI2(aug_params, split='training', halve_image=False)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=True, shuffle=True, num_workers=8, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

