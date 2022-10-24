import torch
import numpy as np
from utils.evaluator import Evaluator
from collections import OrderedDict
class BaseEngine(object):
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.model.to(self.device)
        self.evaluator = Evaluator(5)
    def train(self, train_loader, epoch):
        self.model.train()
        for batch_idx, (data) in enumerate(train_loader):
            data, target = data['image'].to(self.device).float(), data['mask'].to(self.device).long()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    def test(self, test_loader):
        self.model.eval()
        self.evaluator.reset()

        for batch_idx, (data) in enumerate(test_loader):
            data, target = data['image'].to(self.device).float(), data['mask'].to(self.device).long()
            with torch.no_grad():
                output = self.model(data)
            test_loss = self.loss_fn(output, target).item() # sum up batch loss
            
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            
            self.evaluator.add_batch(target, pred)
            
            if batch_idx % 10 == 0:
                print('Test Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), test_loss))

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU, IoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Validation:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, IoU:{}".format(Acc, Acc_class, mIoU, FWIoU, IoU))
        print('Loss: %.3f' % test_loss)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
    def load(self, path):
        data = torch.load(path)
        
        

        new_state_dict = OrderedDict()
        for k, v in data['state_dict'].items():
            if 'last_layer' in k:
                continue
            new_state_dict[k] = v 
        self.model.to(self.device)
        self.model.load_state_dict(new_state_dict, strict=False)