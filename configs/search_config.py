import argparse
import numpy as np

def obtain_search_args():
    parser = argparse.ArgumentParser(description="Pretrain the nas model(super-net)")
    # model
    parser.add_argument('--layers', type=int, default=11, help='supernet layers number')
    parser.add_argument('--depth', type=int, default=4, help='supernet depth')
    parser.add_argument('--input_channel', type=int, default=4, help='input_channel')
    parser.add_argument('--stem_multiplier', type=int, default=64, help='stem_multiplier')
    parser.add_argument('--base_multiplier', type=int, default=64, help='base_multiplier')
    parser.add_argument('--num_classes', type=int, default=25)
    
    # dataset
    parser.add_argument('--dataset', type=str, default='FBP', choices=['FBP'], help='dataset name (default: pascal)')
    NORMALISE_PARAMS = [
                        1.0 / 255,  # SCALE
                        np.array([0.485, 0.456, 0.406, 0.411]).reshape((1, 1, 4)),  # MEAN
                        np.array([0.229, 0.224, 0.225, 0.227]).reshape((1, 1, 4)),  # STD
                        ]
    parser.add_argument("--normalise_params", type=list, default=NORMALISE_PARAMS, help="Normalisation parameters [scale, mean, std],")
    parser.add_argument('--batch_size', type=int, default=2, metavar='N', help='input batch size for training (default: auto)')
    
    # optimizer
    parser.add_argument('--lr', type=float, default=0.025, metavar='LR',help='learning rate (default: auto)')
    args = parser.parse_args()
    
    return args