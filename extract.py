import os
import torch
from models.models_compared import GLU_Net
import argparse
import numpy as np
from utils.pixel_wise_mapping import remap_using_flow_fields
from torch.utils.data import Dataset
import cv2
import warnings
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.neighbors import KDTree
warnings.filterwarnings('ignore')

class Hpatches(Dataset):
    def __init__(self, path):

        self.pair_list = []
        seq_list = sorted(os.listdir(path))
        for seq_idx, seq in enumerate(seq_list):
            im1_path = os.path.join(path, seq, '1.ppm')
            for im_idx in range(2, 7):
                im2_path = os.path.join(path, seq, '%d.ppm' % im_idx)
                H_path = os.path.join(path, seq, 'H_1_%d' % im_idx)
                self.pair_list += [{'name': seq,
                                    'interval': im_idx - 1,
                                    'im1': im1_path,
                                    'im2': im2_path,
                                    'H': H_path}]

    def __getitem__(self, item):
        pair = self.pair_list[item]

        name = pair['name']
        interval = pair['interval']
        im1 = cv2.imread(pair['im1'])
        im2 = cv2.imread(pair['im2'])
        H = np.loadtxt(pair['H'])

        return name, interval, im1, im2, H

    def __len__(self):
        return len(self.pair_list)

def grid_gen(batch_size,ht,wd):
    x = torch.arange(0, wd, 1)
    y = torch.arange(0, ht, 1)
    xx, yy = torch.meshgrid(x, y)
    coord_flow = torch.stack([xx, yy], dim = 2)
    return coord_flow.permute([2, 1, 0]).repeat([batch_size, 1, 1, 1])

def knn_dist_nn(kps1, kps2, coord1, coord2):

    kps1 = kps1.tolist()
    kps2 = kps2.tolist()
    coord1 = coord1.cpu().numpy().tolist()
    coord2 = coord2.cpu().numpy().tolist()

    nn12 = []
    nn21 = []

    tree = KDTree(np.array(kps2))
    for idx, pt in enumerate(np.array(coord1)):
        dist, ind = tree.query([pt], k=1)
        nn12 += ind[0].tolist()

    tree = KDTree(np.array(kps1))
    for idx, pt in enumerate(np.array(coord2)):
        dist, ind = tree.query([pt], k=1)
        nn21 += ind[0].tolist()

    good = []
    pts1 = []
    pts2 = []
    for idx, i in enumerate(nn12):
        if nn21[i] == idx:
            good += [cv2.DMatch(idx, i, 1)]
            pts1 += [kps1[idx]]
            pts2 += [kps2[i]]

    return pts1, pts2, good

def knn_feature_nn(kps1, kps2, coord1, coord2, des1, des2, method='sift', dist=10):
    des1 = np.array(des1)
    des2 = np.array(des2)
    kps1 = kps1.tolist()
    kps2 = kps2.tolist()
    coord1 = coord1.cpu().numpy().tolist()
    coord2 = coord2.cpu().numpy().tolist()

    bf = cv2.BFMatcher()
    tree = KDTree(np.array(kps2))
    neighbor_list1 = []
    for idx, pt in enumerate(np.array(coord1)):

        ind = tree.query_radius([pt], r=dist)
        neighbor_list1 += [ind[0].tolist()]

    tree = KDTree(np.array(kps1))
    neighbor_list2 = []
    for idx, pt in enumerate(np.array(coord2)):
        ind = tree.query_radius([pt], r=dist)
        neighbor_list2 += [ind[0].tolist()]

    nn12 = []
    matches12 = []
    for idx, neighbor in enumerate(neighbor_list1):
        if neighbor == []:
            nn12 += [-1]
            continue
        else:
            if method == 'sift':
                matches = bf.knnMatch(des1[idx:idx + 1], des2[neighbor], k=1)
                matches12 += [cv2.DMatch(idx, neighbor[matches[0][0].trainIdx], 1)]
                nn12 += [neighbor[matches[0][0].trainIdx]]

    nn21 = []
    matches21 = []
    for idx, neighbor in enumerate(neighbor_list2):
        if neighbor == []:
            nn21 += [-1]
            continue
        else:
            if method == 'sift':
                matches = bf.knnMatch(des2[idx:idx + 1], des1[neighbor], k=1)
                matches21 += [cv2.DMatch(idx, neighbor[matches[0][0].trainIdx], 1)]
                nn21 += [neighbor[matches[0][0].trainIdx]]

    good = []
    pts1 = []
    pts2 = []
    for idx, i in enumerate(nn12):
        if i == -1:
            continue
        if nn21[i] == idx:
            good += [cv2.DMatch(idx, i, 1)]
            pts1 += [kps1[idx]]
            pts2 += [kps2[i]]

    return pts1, pts2, good



def flow_consistency(im1, im2, pts, pts_flow, flow21):

    pts = torch.from_numpy(pts).cuda()

    h1, w1, _ = im1.shape
    h2, w2, _ = im2.shape
    pad = 0
    indices = ((pad <= pts_flow[:, 0]) & (pts_flow[:, 0] < w2 - pad) & (pad <= pts_flow[:, 1]) & (
                pts_flow[:, 1] < h2 - pad)).cpu().numpy().tolist()

    pts = pts[indices]
    pts_flow = pts_flow[indices]

    coords_backward = get_match(im2, im1, flow21, pts_flow.cpu().numpy())


    dist = torch.norm(pts - coords_backward, dim=1, p=2)

    threshold = 10
    inliers = dist < threshold

    pts = pts[inliers].cpu().numpy()
    pts_flow = pts_flow[inliers].cpu().numpy()
    dist = dist[inliers].cpu().numpy()

    return pts, pts_flow, dist



def get_match(im1, im2, flow, kps1):
    h, w, _ = im1.shape
    h_, w_, _ = im2.shape
    kp1_norm = kps1.copy()
    kp1_norm[:, 0] = (2 * kp1_norm[:, 0] - w + 1) / w
    kp1_norm[:, 1] = (2 * kp1_norm[:, 1] - h + 1) / h
    kp1_norm = torch.from_numpy(kp1_norm).repeat(1, 1, 1, 1).float().cuda()

    coords = grid_gen(1, h, w).float().cuda()
    coords = coords + flow
    out = F.grid_sample(coords, kp1_norm, align_corners=True)
    coord = out.permute([0, 2, 3, 1]).squeeze()

    coord[:, 0] *= (w_ / w)
    coord[:, 1] *= (h_ / h)

    return coord


def dense_matching(im, flow, type='src2tgt'):

    '''
        type: src2tgt tgt2src
    '''

    grid = grid_gen(1, im.shape[0], im.shape[1]).float().cuda()
    try:
        coord = flow + grid
    except:
        print(flow.shape)
        print(coord.shape)

    if type == 'src2tgt':
        kps1 = grid[0].permute([1,2,0]).reshape(-1,2).cpu().numpy()
        kps2 = coord[0].permute([1,2,0]).reshape(-1,2).cpu().numpy()
    else:
        kps1 = coord[0].permute([1,2,0]).reshape(-1,2).cpu().numpy()
        kps2 = grid[0].permute([1,2,0]).reshape(-1,2).cpu().numpy()
    return kps1, kps2, []


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test GLUNet on a pair of images')

    parser.add_argument('--pre_trained_models_dir', type=str, default='pre_trained_models/',
                        help='Directory containing the pre-trained-models.')
    parser.add_argument('--pre_trained_model', type=str, default='DPED_CityScape_ADE',
                        help='Name of the pre-trained-model.')
    parser.add_argument('--method', type=str, default='glu_feat_nn')

    args = parser.parse_args()

    torch.cuda.empty_cache()
    torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # either gpu or cpu


    with torch.no_grad():
        network = GLU_Net(path_pre_trained_models=args.pre_trained_models_dir,
                          model_type=args.pre_trained_model,
                          consensus_network=False,
                          cyclic_consistency=True,
                          iterative_refinement=True,
                          apply_flipping_condition=False)

    hpatches_path = '/home/panxk/Dataset/Hpatches/hpatches-sequences-release'
    dataset = Hpatches(hpatches_path)

    sift = cv2.xfeatures2d.SIFT_create()

    method = args.method

    print('method',method)

    for (name, interval, im1, im2, H) in tqdm(dataset):

        if name[0] == 'i':
            continue

        h, w = im2.shape[:2]
        im1 = cv2.resize(im1, (w, h))

        kp1, des1 = sift.detectAndCompute(im1, None)
        kp2, des2 = sift.detectAndCompute(im2, None)

        im1_tensor = torch.from_numpy(im1[None]).permute([0, 3, 1, 2])
        im2_tensor = torch.from_numpy(im2[None]).permute([0, 3, 1, 2])

        pts1 = np.array([kp1[idx].pt for idx in range(len(kp1))])
        pts2 = np.array([kp2[idx].pt for idx in range(len(kp2))])

        ''' this flow is from img2 to img1'''
        flow21 = network.estimate_flow(im1_tensor, im2_tensor, device, mode='channel_first')
        flow12 = network.estimate_flow(im2_tensor, im1_tensor, device, mode='channel_first')

        coord1 = get_match(im1, im2, flow12, pts1)
        coord2 = get_match(im2, im1, flow21, pts2)

        keypoints1, keypoints2, score= None, None, None

        if method == 'glu_feat_nn':
            keypoints1, keypoints2, good = knn_feature_nn(pts1, pts2, coord1, coord2, des1, des2, 'sift', 10)
        elif method == 'glu_dist_nn':
            keypoints1, keypoints2, good = knn_dist_nn(pts1, pts2, coord1, coord2)
        elif method == 'glu_sift_cycle':
            keypoints1, keypoints2, score = flow_consistency(im1, im2, pts1, coord1, flow21)
        elif method == 'glu_dense':
            keypoints1, keypoints2, _ = dense_matching(im2, flow21, 'tgt2src')
        else:
            raise NotImplementedError

        save_path = os.path.join(hpatches_path, name, '%d.ppm' % (interval + 1))
        with open( save_path + '.' + method, 'wb') as output_file:
            np.savez(output_file, keypoints1=keypoints1, keypoints2=keypoints2, score=score)
