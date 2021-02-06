import cv2
import os
import os.path as osp
import numpy as np


def normalize(pts):
    mean_x = pts[:, 0].mean()
    mean_y = pts[:, 1].mean()
    pts[:, 0] -= mean_x
    pts[:, 1] -= mean_y

    meanDev_x = np.abs(pts[:, 0]).mean()
    meanDev_y = np.abs(pts[:, 1]).mean()

    pts[:, 0] /= meanDev_x
    pts[:, 1] /= meanDev_y

    return pts


def compute_F(path):

    F_list = []

    kitti_flow_path = path
    image_list = sorted(os.listdir(osp.join(kitti_flow_path, 'image_2')))

    pair_list = [[osp.join(kitti_flow_path, 'image_2', image_list[idx * 2]),
                  osp.join(kitti_flow_path, 'image_2', image_list[idx * 2 + 1])] for idx in range(len(image_list) // 2)]

    sift = cv2.xfeatures2d.SIFT_create()

    for idx, pair in enumerate(pair_list):

        print("%d/%d\r" % (idx, len(pair_list)), end=' ')

        im1 = cv2.cvtColor(cv2.imread(pair[0]), cv2.COLOR_BGR2RGB)
        im2 = cv2.cvtColor(cv2.imread(pair[1]), cv2.COLOR_BGR2RGB)

        kps1, des1 = sift.detectAndCompute(im1, None)
        kps2, des2 = sift.detectAndCompute(im2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        pts1 = []
        pts2 = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
                pts1.append(kps1[m.queryIdx].pt)
                pts2.append(kps2[m.trainIdx].pt)

        pts1 = np.array(pts1)
        pts2 = np.array(pts2)

        pts1_norm = normalize(pts1.copy())
        pts2_norm = normalize(pts2.copy())

        F, mask = cv2.findFundamentalMat(pts1_norm, pts2_norm, cv2.FM_LMEDS)
        F_list += [F.flatten()]

    F_list = np.array(F_list)

    np.savetxt('./F_list.txt', F_list, fmt='%.5f')
    print("F_list saved!")


if __name__ == '__main__':

    kitti_flow_path = '../../datasets/KITTI_Flow/training/'
    compute_F(kitti_flow_path)

