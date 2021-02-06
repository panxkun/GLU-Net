import sys
import os



if __name__ == '__main__':

    sample_path = '/home/panxk/myWork/temp/raft_pose_v2/sample/'
    save_path = '/home/panxk/myWork/GLU-Net/evaluation/'

    scene_list = os.listdir(sample_path)
    for scene in scene_list:
        if os.path.isdir(os.path.join(sample_path, scene)) == False:
            continue
        if scene != 'Hpatches':
            continue
        img_list = os.listdir(os.path.join(sample_path, scene))
        length = len(img_list) // 2
        shuff = img_list[0][-3:]
        for idx in range(length):
            im1_path = os.path.join(sample_path, scene, '%d.%s' % (idx * 2, shuff))
            im2_path = os.path.join(sample_path, scene, '%d.%s' % (idx * 2 + 1, shuff))

            os.system('python test_GLUNet.py --path_source_image %s --path_target_image %s --write_dir %s --img_name %s_%d.png'
                      % (im1_path, im2_path, save_path, scene, idx))

        break
    break

# python test_GLUNet.py --path_source_image  --path_target_image %s --write_dir %s --img_name %s_%d.png