import numpy as np
import argparse
import time
import random
import os
from os import path as osp
from termcolor import colored
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from datasets.training_dataset import HomoAffTps_Dataset
from datasets.load_pre_made_dataset import PreMadeDataset
from utils_training.optimize_GLUNet_with_adaptive_resolution import train_epoch, validate_epoch
from models.our_models.GLUNet import GLUNet_model
from utils_training.utils_CNN import load_checkpoint, save_checkpoint, boolean_string
from tensorboardX import SummaryWriter
from utils.image_transforms import ArrayToTensor

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='GLU-Net train script')
    # Paths
    parser.add_argument('--name_exp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment to save')
    parser.add_argument('--pre_loaded_training_dataset', default=False, type=boolean_string,
                        help='Synthetic training dataset is already created and saved in disk ? default is False')
    parser.add_argument('--training_data_dir', type=str,
                        help='path to directory containing original images for training if --pre_loaded_training_'
                             'dataset is False or containing the synthetic pairs of training images and their '
                             'corresponding flow fieltrain_lossds if --pre_loaded_training_dataset is True')
    parser.add_argument('--evaluation_data_dir', type=str,
                        help='path to directory containing original images for validation if --pre_loaded_training_'
                             'dataset is False or containing the synthetic pairs of validation images and their '
                             'corresponding flow fields if --pre_loaded_training_dataset is True')
    parser.add_argument('--snapshots', type=str, default='./snapshots')
    parser.add_argument('--pretrained', dest='pretrained', default=None,
                       help='path to pre-trained model')
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--momentum', type=float,
                        default=4e-4, help='momentum constant')
    parser.add_argument('--start_epoch', type=int, default=-1,
                        help='start epoch')
    parser.add_argument('--n_epoch', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=8,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--weight-decay', type=float, default=4e-4,
                        help='weight decay constant')
    parser.add_argument('--div_flow', type=float, default=1.0,
                        help='div flow')
    parser.add_argument('--seed', type=int, default=1986,
                        help='Pseudo-RNG seed')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # datasets, pre-processing of the images is done within the network function !
    source_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    target_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])

    if not args.pre_loaded_training_dataset:
        # training dataset, created on the fly at each epoch
        pyramid_param = [520] # means that we get the ground-truth flow field at this size
        train_dataset = HomoAffTps_Dataset(image_path=args.training_data_dir,
                                           csv_file=osp.join('datasets', 'csv_files',
                                                         'homo_aff_tps_train_DPED_CityScape_ADE.csv'),
                                           transforms=source_img_transforms,
                                           transforms_target=target_img_transforms,
                                           pyramid_param=pyramid_param,
                                           get_flow=True,
                                           output_size=(520, 520))

        # validation dataset
        pyramid_param = [520]
        val_dataset = HomoAffTps_Dataset(image_path=args.evaluation_data_dir,
                                         csv_file=osp.join('datasets', 'csv_files',
                                                           'homo_aff_tps_test_DPED_CityScape_ADE.csv'),
                                         transforms=source_img_transforms,
                                         transforms_target=target_img_transforms,
                                         pyramid_param=pyramid_param,
                                         get_flow=True,
                                         output_size=(520, 520))

    else:
        # If synthetic pairs were already created and saved to disk, run instead of 'train_dataset' the following.
        # and replace args.training_data_dir by the root to folders containing images/ and flow/

        flow_transform = transforms.Compose([ArrayToTensor()]) # just put channels first and put it to float
        train_dataset, _ = PreMadeDataset(root=args.training_data_dir,
                                          source_image_transform=source_img_transforms,
                                          target_image_transform=target_img_transforms,
                                          flow_transform=flow_transform,
                                          co_transform=None,
                                          split=1)  # only training

        _, val_dataset = PreMadeDataset(root=args.evaluation_data_dir,
                                        source_image_transform=source_img_transforms,
                                        target_image_transform=target_img_transforms,
                                        flow_transform=flow_transform,
                                        co_transform=None,
                                        split=0)  # only validation

    # Dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.n_threads)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.n_threads)


    # models
    '''
    Default GLU-Net parameters:
    model = GLUNet_model(batch_norm=True, pyramid_type='VGG',
                         div=args.div_flow, evaluation=False,
                         consensus_network=False,
                         cyclic_consistency=True,
                         dense_connection=True,
                         decoder_inputs='corr_flow_feat',
                         refinement_at_all_levels=False,
                         refinement_at_adaptive_reso=True)
    
    
    For SemanticGLU-Net:
    model = SemanticGLUNet_model(batch_norm=True, pyramid_type='VGG',
                             div=args.div_flow, evaluation=False,
                             cyclic_consistency=False, consensus_network=True)
    
    One can change the parameters 
    
    '''
    model = GLUNet_model(batch_norm=True, pyramid_type='VGG',
                         div=args.div_flow, evaluation=False,
                         consensus_network=False,
                         cyclic_consistency=True,
                         dense_connection=True,
                         decoder_inputs='corr_flow_feat',
                         refinement_at_all_levels=False,
                         refinement_at_adaptive_reso=True)
    print(colored('==> ', 'blue') + 'GLU-Net created.')

    # Optimizer
    optimizer = \
        optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                   lr=args.lr,
                   weight_decay=args.weight_decay)

    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[65, 75, 95],
                                         gamma=0.5)
    weights_loss_coeffs = [0.32, 0.08, 0.02, 0.01]

    if args.pretrained:
        # reload from pre_trained_model
        model, optimizer, scheduler, start_epoch, best_val = load_checkpoint(model, optimizer, scheduler,
                                                                 filename=args.pretrained)
        # now individually transfer the optimizer parts...
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        cur_snapshot = os.path.basename(os.path.dirname(args.pretrained))
    else:
        if not os.path.isdir(args.snapshots):
            os.mkdir(args.snapshots)

        cur_snapshot = args.name_exp
        if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
            os.makedirs(osp.join(args.snapshots, cur_snapshot))

        with open(osp.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
            pickle.dump(args, f)

        best_val = float("inf")
        start_epoch = 0

    # create summary writer
    save_path = osp.join(args.snapshots, cur_snapshot)
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))

    model = nn.DataParallel(model)
    model = model.to(device)

    train_started = time.time()

    for epoch in range(start_epoch, args.n_epoch):
        scheduler.step()
        print('starting epoch {}:  learning rate is {}'.format(epoch, scheduler.get_lr()[0]))

        # Training one epoch
        train_loss = train_epoch(model,
                                 optimizer,
                                 train_dataloader,
                                 device,
                                 epoch,
                                 train_writer,
                                 div_flow=args.div_flow,
                                 save_path=os.path.join(save_path, 'train'),
                                 loss_grid_weights=weights_loss_coeffs)


        train_writer.add_scalar('train loss', train_loss, epoch)
        train_writer.add_scalar('learning_rate', scheduler.get_lr()[0], epoch)
        print(colored('==> ', 'green') + 'Train average loss:', train_loss)

        # Validation
        val_loss_grid, val_mean_epe, val_mean_epe_H_8, val_mean_epe_32, val_mean_epe_16 = \
            validate_epoch(model, val_dataloader, device, epoch=epoch,
                           save_path=os.path.join(save_path, 'test'),
                           div_flow=args.div_flow,
                           loss_grid_weights=weights_loss_coeffs)
        print(colored('==> ', 'blue') + 'Val average grid loss :',
              val_loss_grid)
        print('mean EPE is {}'.format(val_mean_epe))
        print('mean EPE from reso H/8 is {}'.format(val_mean_epe_H_8))
        print('mean EPE from reso 32 is {}'.format(val_mean_epe_32))
        print('mean EPE from reso 16 is {}'.format(val_mean_epe_16))
        test_writer.add_scalar('validation images: mean EPE ', val_mean_epe, epoch)
        test_writer.add_scalar('validation images: mean EPE_from_reso_H_8', val_mean_epe_H_8, epoch)
        test_writer.add_scalar('validation images: mean EPE_from_reso_32', val_mean_epe_32, epoch)
        test_writer.add_scalar('validation images: mean EPE_from_reso_16', val_mean_epe_16, epoch)
        test_writer.add_scalar('validation images: val loss', val_loss_grid, epoch)
        print(colored('==> ', 'blue') + 'finished epoch :', epoch + 1)

        # save checkpoint for each epoch and a fine called best_model so far
        is_best = val_mean_epe < best_val
        best_val = min(val_mean_epe, best_val)
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.module.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'best_loss': best_val},
                        is_best, save_path, 'epoch_{}.pth'.format(epoch + 1))

    print(args.seed, 'Training took:', time.time()-train_started, 'seconds')
