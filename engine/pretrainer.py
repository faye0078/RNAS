import torch
import tqdm
from engine.baseEngine import BaseEngine
from model.encoder import modelEncoder
from dataloaders import make_data_loader
class Pretrainer(BaseEngine):
    """_summary_

    Args:
        BaseEngine (_type_): _description_
    """
    def __init__(self, model, optimizer, loss_fn, device, layers, depth):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load("./pretrain_model.pth"))
        self.model = torch.nn.DataParallel(self.model)
        self.model_encoder = modelEncoder(layers, depth)
        
    def pretrain_one(self, train_loader, epoch):
        print("Epoch: ", epoch)
        tbar = tqdm.tqdm(train_loader, ncols=80)
        for i, sample in enumerate(tbar):
            image = sample['image'].to(self.device).float()
            mask = sample['mask'].to(self.device).long()
            output = self.model(image)
            loss = self.loss_fn(output, mask)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            tbar.set_description('Train loss: %.3f' % loss.item())
            # if i > 3:
            #     break
        torch.cuda.empty_cache()
            
    def update_model_encode(self):
        self.model.module.update_active_encode(self.model_encoder.generate())
