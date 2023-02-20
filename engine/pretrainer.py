import torch
from engine.baseEngine import BaseEngine
from model.encoder import modelEncoder
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
        self.model_encoder = modelEncoder(layers, depth)
        
    def pretrain(self, model_size, img_size):
        self.model_size = model_size
        self.img_size = img_size