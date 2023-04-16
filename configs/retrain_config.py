import argparse
import numpy as np

def obtain_retrain_args():
    parser = argparse.ArgumentParser(description="ReTrain the nas model")
    
    # model
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='test', help='set the checkpoint name')
    parser.add_argument('--model_name', type=str, default='RNAS', choices=['RNAS', 'DNAS', 'hrnet', 'deeplabv3plus', 'pspnet', 'unet', 'refinenet', 'fast-nas', 'SrNet', 'one_path','MACUNet', 'MAResUNet', 'MSFCN'], help='the model name')
    parser.add_argument('--layers', type=int, default=11, help='supernet layers number')
    parser.add_argument('--depth', type=int, default=4, help='supernet depth')
    parser.add_argument('--stem_multiplier', type=int, default=64, help='stem_multiplier')
    parser.add_argument('--base_multiplier', type=int, default=64, help='base_multiplier')

    parser.add_argument('--model_encode_path', type=str, default=None)
    parser.add_argument('--cell_arch', type=str, default=None)
    parser.add_argument('--affine', default=False, type=bool, help='whether use affine in BN')
    
    # data
    parser.add_argument('--batch_size', type=int, default=2, metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--dataset', type=str, default='normal_FBP', choices=['normal_FBP', 'pascal', 'coco', 'cityscapes', 'kd', 'GID', 'hps-GID', 'uadataset', 'uadataset_dfc'], help='dataset name (default: pascal)')
    parser.add_argument('--crop_size', type=int, default=512, help='crop image size')
    parser.add_argument('--resize', type=int, default=512, help='resize image size')
    NORMALISE_PARAMS = [
                        1.0 / 255,  # SCALE
                        np.array([0.485, 0.456, 0.406, 0.411]).reshape((1, 1, 4)),  # MEAN
                        np.array([0.229, 0.224, 0.225, 0.227]).reshape((1, 1, 4)),  # STD
                        ]
    parser.add_argument("--normalise_params", type=list, default=NORMALISE_PARAMS, help="Normalisation parameters [scale, mean, std],")
    parser.add_argument('--nclass', type=int, default=5, help='number of class')
    parser.add_argument('--num_classes', type=int, default=25)

    # train
    parser.add_argument('--nas', type=str, default='train', choices=['search', 'train'])
    parser.add_argument('--num_worker', type=int, default=4,metavar='N', help='numer workers')
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal'], help='loss func type (default: ce)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N', help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    parser.add_argument('--lr', type=float, default=0.025, metavar='LR', help='learning rate (default: auto)')
    parser.add_argument('--min_lr', type=float, default=0.001)
    parser.add_argument('--lr_scheduler', type=str, default='cos', choices=['poly', 'step', 'cos'], help='lr scheduler mode')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=3e-4, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    args = parser.parse_args()
    return args
