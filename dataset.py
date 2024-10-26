import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import math

class MonteCarloDataset(Dataset):
    """Dataset containing RGB, Normal, Depth and Albedo buffers of Monte Carlo renders"""

    def __init__(self, root_dir, subseq_len, transform=None):
        """
        Args:
            root_dir (string): Directory containing image subdirectories 
                (root_dir/rgb, root_dir/normal, root_dir/depth, root_dir/albedo)
        """
        self.root_dir = root_dir
        self.subseq_len = subseq_len
        self.frames_total = self.count_frames(root_dir)
        self.num_valid_startpoints = self.frames_total - ((self.subseq_len - 1) * self.num_scenes)
        self.transform = transform

    def __len__(self):
        return self.num_valid_startpoints
    
    def __getitem__(self, idx):
        """Assumes idx is integer in [0, __len__())"""

        if torch.is_tensor(idx):
            idx = idx.tolist()

        scene_idx = idx // (self.frames_per_scene - (self.subseq_len - 1))
        idx += self.subseq_len * scene_idx

        scene = self.scene_names[scene_idx]

        frames = np.array([ x for x in range(idx, idx + self.subseq_len) ]) + 1
        frames = frames % (self.frames_per_scene + 1)
        frames = [ str(f).zfill(5) for f in frames ]

        samples = torch.randint(low=0, high=10, size=(self.subseq_len,)).tolist()
        samples = [ str(s).zfill(3) for s in samples ]

        datatype = np.float32

        albedo = np.empty((self.subseq_len, 1024, 1024, 3), dtype=datatype)
        depth = np.empty((self.subseq_len, 1024, 1024, 1), dtype=datatype)
        normal = np.empty((self.subseq_len, 1024, 1024, 3), dtype=datatype)
        rgb_gt = np.empty((self.subseq_len, 1024, 1024, 3), dtype=datatype)
        rgb_1spp = np.empty((self.subseq_len, 1024, 1024, 3), dtype=datatype)
        motion = np.empty((self.subseq_len, 1024, 1024, 4), dtype=datatype)
        normal_vanilla = np.empty((self.subseq_len, 1024, 1024, 3), dtype=datatype)

        for i, frame in enumerate(frames):
            albedo[i] = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, 'albedo', f'albedo_{scene}_{frame}_gt.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype(datatype)
            depth[i] = np.expand_dims(cv2.imread(os.path.join(self.root_dir, 'depth', f'depth_{scene}_{frame}_gt.exr'), cv2.IMREAD_UNCHANGED)[:, :, 0], axis=-1).astype(datatype)
            normal[i] = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, 'hdr_normal', f'normal_{scene}_{frame}_gt.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype(datatype)
            rgb_gt[i] = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, 'rgb', f'rgb_{scene}_{frame}_gt.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype(datatype)
            rgb_1spp[i] = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, 'rgb', f'rgb_{scene}_{frame}_{samples[i]}.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype(datatype)
            motion[i] = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, 'motion', f'motion_{scene}_{frame}_gt.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA).astype(datatype)
            normal_vanilla[i] = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, 'normal', f'normal_{scene}_{frame}_gt.png'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype(datatype)

        frame = {'albedo': albedo, 
                 'depth': depth, 
                 'normal': normal, 
                 'rgb_gt': rgb_gt, 
                 'rgb_1spp': rgb_1spp,
                 'motion': motion,
                 'normal_vanilla': normal_vanilla}

        if self.transform:
            frame = self.transform(frame)

        return frame
    
    def count_frames(self, root_dir):
        self.scene_names = set()
        max_frame = -1

        for fname in os.listdir(os.path.join(root_dir, 'rgb')):
            split_fname = fname.split('.')[0].split('_')
            self.scene_names.add('_'.join(split_fname[1:len(split_fname) - 2]))
            max_frame = max(int(split_fname[-2]), max_frame)

        self.num_scenes = len(self.scene_names)
        self.frames_per_scene = max_frame

        self.scene_names = list(self.scene_names)

        return self.frames_per_scene * self.num_scenes
    
class MonteCarloTestset(Dataset):
    def __init__(self, root_dir, width, height, transform=None):
        self.root_dir = root_dir
        self.frames_total = self.count_frames(root_dir)
        self.transform = transform
        self.w = width
        self.h = height

    def __len__(self):
        return self.frames_total
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        scene_idx = idx // self.frames_per_scene
        scene = self.scene_names[scene_idx]

        frame = str((idx % self.frames_per_scene) + 1).zfill(5)

        datatype = np.float32
        albedo = np.empty((1, self.h, self.w, 3), dtype=datatype)
        depth = np.empty((1, self.h, self.w, 1), dtype=datatype)
        normal = np.empty((1, self.h, self.w, 3), dtype=datatype)
        rgb_gt = np.empty((1, self.h, self.w, 3), dtype=datatype)
        rgb_1spp = np.empty((1, self.h, self.w, 3), dtype=datatype)
        motion = np.empty((1, self.h, self.w, 4), dtype=datatype)
        normal_vanilla = np.empty((1, self.h, self.w, 3), dtype=datatype)

        albedo[0] = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, 'albedo', f'albedo_{scene}_{frame}_gt.exr'), cv2.IMREAD_UNCHANGED)[:self.h, :self.w], cv2.COLOR_BGR2RGB).astype(datatype)
        depth[0] = np.expand_dims(cv2.imread(os.path.join(self.root_dir, 'depth', f'depth_{scene}_{frame}_gt.exr'), cv2.IMREAD_UNCHANGED)[:self.h, :self.w, 0], axis=-1).astype(datatype)
        normal[0] = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, 'hdr_normal', f'normal_{scene}_{frame}_gt.exr'), cv2.IMREAD_UNCHANGED)[:self.h, :self.w], cv2.COLOR_BGR2RGB).astype(datatype)
        rgb_gt[0] = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, 'rgb', f'rgb_{scene}_{frame}_gt.exr'), cv2.IMREAD_UNCHANGED)[:self.h, :self.w], cv2.COLOR_BGR2RGB).astype(datatype)
        rgb_1spp[0] = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, 'rgb', f'rgb_{scene}_{frame}_000.exr'), cv2.IMREAD_UNCHANGED)[:self.h, :self.w], cv2.COLOR_BGR2RGB).astype(datatype)
        motion[0] = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, 'motion', f'motion_{scene}_{frame}_gt.exr'), cv2.IMREAD_UNCHANGED)[:self.h, :self.w], cv2.COLOR_BGRA2RGBA).astype(datatype)
        normal_vanilla[0] = cv2.cvtColor(cv2.imread(os.path.join(self.root_dir, 'normal', f'normal_{scene}_{frame}_gt.png'), cv2.IMREAD_UNCHANGED)[:self.h, :self.w], cv2.COLOR_BGR2RGB).astype(datatype)

        frame = {'albedo': albedo, 
                 'depth': depth, 
                 'normal': normal, 
                 'rgb_1spp': rgb_1spp,
                 'rgb_gt': rgb_gt, 
                 'motion': motion,
                 'normal_vanilla': normal_vanilla,
                 'rgb_in': rgb_1spp.copy()}

        if self.transform:
            frame = self.transform(frame)

        frame['scene'] = scene

        return frame
    
    def count_frames(self, root_dir):
        self.scene_names = set() 
        max_frame = -1

        for fname in os.listdir(os.path.join(root_dir, 'rgb')):
            split_fname = fname.split('.')[0].split('_')
            scene_name = '_'.join(split_fname[1:len(split_fname) - 2])
            self.scene_names.add(scene_name)

            frame_number = int(split_fname[-2])
            max_frame = max(frame_number, max_frame)

        self.num_scenes = len(self.scene_names)

        self.frames_per_scene = max_frame

        self.scene_names = list(self.scene_names)

        return self.frames_per_scene * self.num_scenes