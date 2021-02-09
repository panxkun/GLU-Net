import sys
sys.path.append('../')
import numpy as np
import torch
import torch.utils.data as data
import os
import random
import cv2
import time
from datasets.transformation import SynthPairTnf, Theta_gen


def refine_grid(grid):
    H, W, _ = grid.shape
    grid[:,:,0] = (grid[:,:,0] + 1/2) * W
    grid[:,:,1] = (grid[:,:,1] + 1/2) * H

    return grid

def fundamental_matrix_gen(T1, T2, K1, K2, proj1, proj2, s=None):

    Twc1 = proj1.dot(T1)
    Twc2 = proj2.dot(T2)

    T21 = np.linalg.inv(Twc2).dot(Twc1)
    R = T21[:3, :3]
    t = T21[:3, 3]

    t_antisymmetric = np.zeros([3, 3])
    t_antisymmetric[0][1] = -t[2]
    t_antisymmetric[0][2] = t[1]
    t_antisymmetric[1][0] = t[2]
    t_antisymmetric[1][2] = -t[0]
    t_antisymmetric[2][0] = -t[1]
    t_antisymmetric[2][1] = t[0]

    essential_matrix = t_antisymmetric.dot(R)

    H, W = 480, 640
    size_list = [[H//8, W//8], [H//4, W//4], [16, 16], [32, 32]]

    F_list = []
    for size in size_list:
        s = np.diag([size[1] / W, size[0] / H, 1])
        K1s = s.dot(K1)
        K2s = s.dot(K2)
        F = (np.linalg.inv(K2s.transpose()).dot(essential_matrix)).dot(np.linalg.inv(K1s))
        F_list.append(F)

    return F_list

def spatial_transform(dataset, img1, img2, pose1, pose2,
                      intrinsic1, intrinsic2, proj1, proj2, crop_size):

    ht, wd = img1.shape[:2]
    size_x = int(crop_size[1] // 8 * 8)
    size_y = int(crop_size[0] // 8 * 8)

    assert (size_x % 8 == 0 and size_y % 8 == 0)

    img1 = cv2.resize(img1, (size_x, size_y), interpolation=cv2.INTER_LINEAR)
    img2 = cv2.resize(img2, (size_x, size_y), interpolation=cv2.INTER_LINEAR)

    s = np.diag([size_x / wd, size_y / ht, 1.0])

    K1 = intrinsic1
    K2 = intrinsic2

    if dataset == 'KITTI':
        pose1 = np.concatenate((pose1, np.array([[0., 0., 0., 1.]])), axis=0)
        pose2 = np.concatenate((pose2, np.array([[0., 0., 0., 1.]])), axis=0)

    if dataset == 'MegaDepth':
        pose1 = np.linalg.inv(pose1)
        pose2 = np.linalg.inv(pose2)

    F = fundamental_matrix_gen(pose1, pose2, K1, K2, proj1, proj2, s)

    img1 = np.ascontiguousarray(img1)
    img2 = np.ascontiguousarray(img2)
    F = np.ascontiguousarray(F)

    return img1, img2, F


class PoseDataset(data.Dataset):
    def __init__(self,
                 root,
                 source_image_transform=None,
                 target_image_transform=None,
                 ):

        self.pair_list = np.load(root, allow_pickle=True)
        self.data_mode = 'pair'
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.init_seed = False

        h, w = 480, 640
        tnf_list = ['affine', 'hom', 'tps']
        self.pair_generation_tnf = [SynthPairTnf(geometric_model=tnf,
                                                output_size=(h, w),
                                                use_cuda=False) for tnf in tnf_list]

        self.theta_gen = [Theta_gen(geometric_model=tnf,
                                   output_size=(h, w),) for tnf in tnf_list]

    def __getitem__(self, index):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.pair_list)

        img1 = cv2.imread(self.pair_list[index][0]['im'])
        img2 = cv2.imread(self.pair_list[index][1]['im'])
        intrinsic1 = self.pair_list[index][0]['intrinsic']
        intrinsic2 = self.pair_list[index][1]['intrinsic']
        proj1 = self.pair_list[index][0]['proj']
        proj2 = self.pair_list[index][1]['proj']
        pose1 = self.pair_list[index][0]['pose']
        pose2 = self.pair_list[index][1]['pose']

        try:
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)
        except Exception as e:
            print(self.pair_list[index][0]['im'], self.pair_list[index][1]['im'])
            raise e

        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        crop_size = img1.shape[:2]

        img1, img2, F = spatial_transform('MegaDepth', img1, img2, pose1, pose2,
                                                       intrinsic1, intrinsic2, proj1, proj2, crop_size)

        if self.source_image_transform is not None:
            img1 = self.source_image_transform(img1)
        if self.target_image_transform is not None:
            img2 = self.target_image_transform(img2)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        F = torch.from_numpy(F).float()

        tnf_index = np.random.randint(3)

        batch = {'image': None, 'theta': None}
        batch['image'] = img2[None]
        batch['theta'] = self.theta_gen[tnf_index]()[None]
        tnf = self.pair_generation_tnf[tnf_index](batch)
        img3 = tnf['target_image'][0]

        grid = refine_grid(tnf['warped_grid'][0])


        H, W = grid.shape[:2]

        coords = torch.meshgrid(torch.arange(H), torch.arange(W))
        coords = torch.stack(coords[::-1], dim=0).float().permute([1,2,0])

        grid = grid - coords


        return {'source_image': img2,
                'target_image': img3,
                'flow_map': grid,
                }


        # return {'source_image': img1,
        #         'target_image': img2,
        #         'F': F
        #         }


    def __len__(self):
        return len(self.pair_list)