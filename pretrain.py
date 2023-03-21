# train the supernet
import argparse
import torch
torch.cuda.set_enabled_lms(True)
from configs.pretrain_config import obtain_pretrain_args
from engine.pretrainer import Pretrainer
from model.supernet import SuperNet
from dataloaders import make_data_loader
def main():
    # init the pretrainer
    args = obtain_pretrain_args()
    model = SuperNet(args.layers, args.depth, args.input_channel, args.num_classes, args.stem_multiplier, args.base_multiplier)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    pretrainer = Pretrainer(model, optimizer, loss_fn, device, args.layers, args.depth)
    
    for epoch in range(args.epochs):
        pretrainer.update_model_encode()
        pretrainer.pretrain_one(make_data_loader(args)[0], epoch)

if __name__ == '__main__':
    main()