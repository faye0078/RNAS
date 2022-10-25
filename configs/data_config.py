import argparse
import numpy as np

def obtain_data_args():
    parser = argparse.ArgumentParser(description="Only Get the Data Args")
    
    # data
    parser.add_argument('--batch_size', type=int, default=12, metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--dataset', type=str, default='GID', choices=['pascal', 'coco', 'cityscapes', 'kd', 'GID', 'hps-GID', 'uadataset', 'uadataset_dfc'], help='dataset name (default: pascal)')
    parser.add_argument('--crop_size', type=int, default=512, help='crop image size')
    parser.add_argument('--resize', type=int, default=512, help='resize image size')
    NORMALISE_PARAMS = [
                        1.0 / 255,  # SCALE
                        np.array([0.485, 0.456, 0.406, 0.411]).reshape((1, 1, 4)),  # MEAN
                        np.array([0.229, 0.224, 0.225, 0.227]).reshape((1, 1, 4)),  # STD
                        ]
    parser.add_argument("--normalise_params", type=list, default=NORMALISE_PARAMS, help="Normalisation parameters [scale, mean, std],")
    parser.add_argument('--nclass', type=int, default=5, help='number of class')
    

    args = parser.parse_args()
    return args