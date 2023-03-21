import argparse
import numpy as np

def obtain_pretrain_args():
    parser = argparse.ArgumentParser(description="Pretrain the nas model(super-net)")
    
    # model
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='test', help='set the checkpoint name')
    parser.add_argument('--model_name', type=str, default='SuperNet', choices=['SuperNet'], help='set the model name')
    parser.add_argument('--model_size', type=int, default=1000)
    parser.add_argument('--layers', type=int, default=11, help='supernet layers number')
    parser.add_argument('--depth', type=int, default=4, help='supernet depth')
    parser.add_argument('--input_channel', type=int, default=4, help='input_channel')
    parser.add_argument('--stem_multiplier', type=int, default=64, help='stem_multiplier')
    parser.add_argument('--base_multiplier', type=int, default=64, help='base_multiplier')
    parser.add_argument('--affine', default=False, type=bool, help='whether use affine in BN')
    # data
    parser.add_argument('--batch_size', type=int, default=2, metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--dataset', type=str, default='FBP', choices=['FBP'], help='dataset name (default: pascal)')
    NORMALISE_PARAMS = [
                        1.0 / 255,  # SCALE
                        np.array([0.485, 0.456, 0.406, 0.411]).reshape((1, 1, 4)),  # MEAN
                        np.array([0.229, 0.224, 0.225, 0.227]).reshape((1, 1, 4)),  # STD
                        ]
    parser.add_argument('--num_classes', type=int, default=25)
    parser.add_argument("--normalise_params", type=list, default=NORMALISE_PARAMS, help="Normalisation parameters [scale, mean, std],")
    
    # train
    parser.add_argument("--train_stage", type=str, default='pretrain', choices=['pretrain', 'finetune'], help="train stage")
    parser.add_argument('--num_workers', type=int, default=8,metavar='N', help='dataloader threads')
    parser.add_argument('--epochs', type=int, default=60, metavar='N', help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal'], help='loss func type (default: ce)')
    parser.add_argument('--lr', type=float, default=0.025, metavar='LR',help='learning rate (default: auto)')
    parser.add_argument('--min_lr', type=float, default=0.001)
    parser.add_argument('--lr_scheduler', type=str, default='cos', choices=['poly', 'step', 'cos'], help='lr scheduler mode')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=3e-4, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()
    return args