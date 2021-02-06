import sys
sys.path.append('../../core/')
sys.path.append('../../tool/')
sys.path.append('../../')
import argparse
import numpy as np
import torch
import datasets
from models.models_compared import GLU_Net
import cv2
import os
import torch.utils.data as data
import warnings
warnings.filterwarnings('ignore')

def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid


class KITTI_Flow(data.Dataset):
    def __init__(self, path, type='occ'):
        self.pair_list = []
        self.type = type

        image_list = sorted(os.listdir(os.path.join(path, 'image_2')))
        flow_list = sorted(os.listdir(os.path.join(path, 'flow_%s' % type)))

        try:
            F_list = np.loadtxt('./F_list.txt')
        except:
            F_list = np.loadtxt('experiment/KITTI_Flow/F_list.txt')

        for idx in range(len(image_list)//2):
            im1_path = os.path.join(path, 'image_2', image_list[idx * 2])
            im2_path = os.path.join(path, 'image_2', image_list[idx * 2 + 1])
            flow_path = os.path.join(path, 'flow_%s' % type, flow_list[idx])
            F = F_list[idx]

            self.pair_list += [{'id': idx,
                                'im1': im1_path,
                                'im2': im2_path,
                                'flow': flow_path,
                                'F': F}]

    def __getitem__(self, item):

        pair = self.pair_list[item]

        im1 = cv2.imread(pair['im1'])
        im2 = cv2.imread(pair['im2'])
        flow_gt, valid_gt = readFlowKITTI(pair['flow'])
        F = pair['F'].reshape(3,3)

        return im1, im2, flow_gt, valid_gt, F

    def __len__(self):

        return len(self.pair_list)


class Evaluate_Flow:
    def __init__(self, args, model, type='occ', online=True):
        self.args = args
        self.dataset = KITTI_Flow(self.args.flow_path, type)
        self.extractor = network

    def evaluate_flow(self, type='occ'):

        out_list, epe_list, dist_cos_list, dist_sin_list = [], [], [], []

        for idx, (im1, im2, flow_gt, valid_gt, F) in enumerate(self.dataset):
            print('\rEvaluate KITTI Flow %s %d/%d' % (self.dataset.type, idx, len(self.dataset)), end='')

            H, W, _ = im1.shape

            im1_tensor = torch.from_numpy(im1[None]).permute([0, 3, 1, 2]).cuda()
            im2_tensor = torch.from_numpy(im2[None]).permute([0, 3, 1, 2]).cuda()
            flow_pr = self.extractor.estimate_flow(im2_tensor, im1_tensor, device, mode='channel_first')[0]


            flow_gt = torch.from_numpy(flow_gt).cuda()
            valid_gt = torch.from_numpy(valid_gt).cuda()


            epe = torch.sum((flow_pr.permute([1, 2, 0]) - flow_gt) ** 2, dim=2).sqrt()
            mag = torch.sum(flow_gt ** 2, dim=2).sqrt()

            '''compute the verticl and parallel'''
            flow_gt_flatten = flow_gt.reshape(-1, 2).cpu().numpy()
            flow_pr_flatten = flow_pr.permute([1, 2, 0]).reshape(-1, 2).cpu().numpy()
            indices = np.where(valid_gt.cpu().numpy().reshape(-1) == True)[0]
            flow_gt_flatten = flow_gt_flatten[indices]
            flow_pr_flatten = flow_pr_flatten[indices]

            x1 = np.array([[idx % W, idx // W] for idx in indices])
            x2_gt = flow_gt_flatten + x1  # gt x2
            x2_pr = flow_pr_flatten + x1  # pr x2

            x1_h = np.concatenate((x1, np.ones([len(indices), 1])), axis=1)
            x2_gt_h = np.concatenate((x2_gt, np.ones([len(indices), 1])), axis=1)
            x2_pr_h = np.concatenate((x2_pr, np.ones([len(indices), 1])), axis=1)

            epipline = F.dot(x1_h.T).T

            vec1 = np.stack((epipline[:,1], -epipline[:,0]), axis=1)
            vec2 = np.stack((x2_pr_h[:,0] - x2_gt_h[:,0], x2_pr_h[:,1] - x2_gt_h[:,1]), axis=1)

            costheta = np.abs(np.sum(vec1*vec2, axis=1)) / (np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1))
            sintheta = np.sqrt(1 - costheta ** 2)
            dist_cos = epe.cpu().numpy().reshape(-1)[indices] * costheta
            dist_sin = epe.cpu().numpy().reshape(-1)[indices] * sintheta

            '''end'''
            epe = epe.view(-1)
            mag = mag.view(-1)
            val = valid_gt.view(-1) >= 0.5

            out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
            epe_list.append(epe[val].mean().item())
            out_list.append(out[val].cpu().numpy())
            dist_cos_list.extend(dist_cos.tolist())
            dist_sin_list.extend(dist_sin.tolist())


        epe_list = np.array(epe_list)
        out_list = np.concatenate(out_list)
        dist_sin_list = np.array(dist_sin_list)
        dist_cos_list = np.array(dist_cos_list)

        epe = np.mean(epe_list)
        f1 = 100 * np.mean(out_list)
        dist_sin = dist_sin_list.mean()
        dist_cos = dist_cos_list.mean()

        print("\rEvaluate KITTI Flow %s | \n"
              "epe: %f\n"
              "f1: %f\n"
              "vertical: %f\n"
              "parallel: %f"
              %(type, epe, f1, dist_sin, dist_cos))

        return epe, f1, dist_sin, dist_cos


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test GLUNet on a pair of images')

    parser.add_argument('--pre_trained_models_dir', type=str, default='pre_trained_models/',
                        help='Directory containing the pre-trained-models.')
    parser.add_argument('--pre_trained_model', type=str, default='DPED_CityScape_ADE',
                        help='Name of the pre-trained-model.')
    parser.add_argument('--method', type=str, default='glu_feat_nn')
    parser.add_argument('--flow_path', type=str, default='/home/panxk/Dataset/KITTI_Flow/training')

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


    exp = Evaluate_Flow(args, network, 'occ', online=False)
    exp.evaluate_flow('occ')
    exp = Evaluate_Flow(args, network, 'noc', online=False)
    exp.evaluate_flow('noc')