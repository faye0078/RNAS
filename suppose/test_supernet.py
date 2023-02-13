# train by stage
import torch
import sys
sys.path.append('/media/dell/DATA/wy/RNAS')
from dataloaders import make_data_loader
from model.seg_hrnet import get_seg_model
from model.supernet import SuperNet
from engine.baseEngine import BaseEngine
from configs.data_config import obtain_data_args

if __name__ == '__main__':
    model = SuperNet(12, 4, )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    engine = BaseEngine(model, optimizer, loss_fn, device)
    
    model_paths = '/media/dell/DATA/wy/RNAS/model/weights/hrnet_512.pth.tar'
    engine.load(model_paths)
    
    args = obtain_data_args()
    kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last':True}
    train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)
    for epoch in range(10):
        engine.train(train_loader, epoch+1)
        
    engine.test(test_loader)
    engine.save('hrnet_stage4_preload.pth.tar')