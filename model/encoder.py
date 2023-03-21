import random
import numpy as np
import torch
from torchstat import stat
from model.supernet import SuperNet
class modelEncoder:
    def __init__(self, layers, depth):
        self.layers = layers
        self.depth = depth
    
    def _get_layer_active(self):
        prob = random.random()
        layer_active = np.zeros((self.depth, self.depth))
        for i in range(self.depth):
            for j in range(self.depth):
                if i == j:
                    layer_active[i][j] = 1
                else:
                    active = random.random()
                    if active > prob:
                        layer_active[i][j] = 1
        return layer_active
    
    def _test_active_model(self, node_active_encode):
        # test the active model
        model = SuperNet(self.active_layers, self.depth, 3, 12, 64, 64, node_active_encode)
        model._initialize_weights()
        device = torch.device("cpu")
        model = model.to(device)
        a = stat(model, (3, 512, 512))
        return a.split('\n')[-5]
        
    def generate(self):
        # generate the model active encode
        active_encode = np.zeros((self.layers, self.depth, self.depth))
        self.active_layers = random.randint(4, self.layers)
        for i in range(self.active_layers):
            active_encode[i] = self._get_layer_active()
        return active_encode
            
if __name__ == '__main__':
    encoder = modelEncoder(12, 4)
    mem_list = []
    for i in range(10):
        node_active_encode = encoder.generate()
        # print(node_active_encode)
        mem = encoder._test_active_model(node_active_encode)
        mem_list.append(mem)
    print(mem_list)
        