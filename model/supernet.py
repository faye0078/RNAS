import torch
import torch.nn as nn
import numpy as np
BatchNorm2d = torch.nn.BatchNorm2d
# BatchNorm2d = torch.nn.SyncBatchNorm
from model.edge import BasicEdge, KeepEdge, DownsampleEdge, UpsampleEdge


class Node(nn.Module):
    def __init__(self, index, base_multiplier):
        """node structure in supernet

        Args:
            index (int): index the node location in depth
            base_multiplier (int): the supernet/node base channel

        Raises:
            ValueError: _description_
        """        
        if index not in [0, 1, 2, 3]:
            raise ValueError(f'index must be in [0, 1, 2, 3], but got {index} the max index is depth - 1, when depth change, "if" condition must be changed')
        super(Node, self).__init__()
        bm = base_multiplier
        if index == 0:
            self.edge0 = KeepEdge(bm, bm)
            self.edge1 = UpsampleEdge(bm*2, bm, 1)
            self.edge2 = UpsampleEdge(bm*4, bm, 2)
            self.edge3 = UpsampleEdge(bm*8, bm, 3)
            self.keep_op = BasicEdge(bm, bm)
        elif index == 1:
            self.edge0 = DownsampleEdge(bm, bm*2, 1)
            self.edge1 = KeepEdge(bm*2, bm*2)
            self.edge2 = UpsampleEdge(bm*4, bm*2, 1)
            self.edge3 = UpsampleEdge(bm*8, bm*2, 2)
            self.keep_op = BasicEdge(2*bm, 2*bm)
        elif index == 2:
            self.edge0 = DownsampleEdge(bm, bm*4, 2)
            self.edge1 = DownsampleEdge(bm*2, bm*4, 1)
            self.edge2 = KeepEdge(bm*4, bm*4)
            self.edge3 = UpsampleEdge(bm*8, bm*4, 1)
            self.keep_op = BasicEdge(4*bm, 4*bm)
        elif index == 3:
            self.edge0 = DownsampleEdge(bm, bm*8, 3)
            self.edge1 = DownsampleEdge(bm*2, bm*8, 2)
            self.edge2 = DownsampleEdge(bm*4, bm*8, 1)
            self.edge3 = KeepEdge(bm*8, bm*8)
            self.keep_op = BasicEdge(8*bm, 8*bm)
            
    def forward(self, x_list, active_encode):
        y = []
        # reduce complexity
        for i in range(4):
            if active_encode[i] and not isinstance(x_list[i], int):
                if i == 0:
                    y.append(self.edge0(x_list[0]))
                elif i == 1:
                    y.append(self.edge1(x_list[1]))
                elif i == 2:
                    y.append(self.edge2(x_list[2]))
                elif i == 3:
                    y.append(self.edge3(x_list[3]))
        if len(y) == 0:
            return 0
        # make y has the same shape
        shape_list = [y[i].shape for i in range(len(y))]
        max_shape = max(shape_list, key=lambda x: x[2])
        for i in range(len(y)):
            if y[i].shape != max_shape:
                y[i] = nn.functional.interpolate(y[i], size=max_shape[2:], mode='bilinear', align_corners=True)
        return self.keep_op(sum(y))

class SuperNet(nn.Module):
    
    def __init__(self, layers, depth, input_channel, num_classes, stem_multiplier, base_multiplier):
        """the main model in this project

        Args:
            layers (int): the number of layers in the model
            depth (int): the depth of the model
            input_channel (int): the number of input channel
            num_classes (int): the number of classes
            stem_multiplier (int): the stem intermediate channel 
            base_multiplier (int): the supernet base channel
        """        
        super(SuperNet, self).__init__()
        self.layers = layers
        self.depth = depth
        
        # stem choice1
        self.stem0 = nn.Sequential(
            nn.Conv2d(input_channel, stem_multiplier, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(stem_multiplier), # default bn settings
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(stem_multiplier, base_multiplier, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(base_multiplier),
            nn.ReLU(inplace=True),
        )
        self.trans0 = KeepEdge(base_multiplier, base_multiplier)
        self.trans1 = DownsampleEdge(base_multiplier, 2*base_multiplier, 1)
        self.trans2 = DownsampleEdge(base_multiplier, 4*base_multiplier, 2)
        self.trans3 = DownsampleEdge(base_multiplier, 8*base_multiplier, 3)
        
        # stem choice2
        # TODO: add stem choice2/ directly downsample the original image and input unique stem
        
        self.node_modules = nn.ModuleList()
        
        self.node_active_encode = np.ones((layers, depth, depth), dtype=bool)

        self.num_connect = np.sum(self.node_active_encode, axis=2)
        
        last_channel_num = 0
        
        for i in range(layers):
            layer_nodes = nn.ModuleList()
            for j in range(depth):
                layer_nodes.append(Node(j, base_multiplier))
                if i == layers -1 and self.num_connect[i][j] != 0:
                     last_channel_num += base_multiplier * pow(2, j)
                     
            self.node_modules.append(layer_nodes)

        self.classifier = nn.Sequential(
            nn.Conv2d(
                in_channels=last_channel_num,
                out_channels=last_channel_num,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_channel_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_channel_num,
                out_channels=num_classes,
                kernel_size=3,
                stride=1,
                padding=1) # TODO: padding=1 or 0 / depends on the kernel size?
        )
        
    def update_active_encode(self, node_active_encode):
        """update the active encode of the model

        Args:
            node_active_encode (array): every node connect condition. shape: (layers, depth, depth)
        """        
        self.node_active_encode = node_active_encode
        self.num_connect = np.sum(self.node_active_encode, axis=2)
        
    def forward(self, x):
        stem0 = self.stem0(x)
        stem1 = self.stem1(stem0)
        
        x_list = []
        x_list.append(self.trans0(stem1))
        x_list.append(self.trans1(stem1))
        x_list.append(self.trans2(stem1))
        x_list.append(self.trans3(stem1))
        
        for i in range(self.layers):
            for j in range(self.depth):
                x_list[j] = self.node_modules[i][j](x_list, self.node_active_encode[i, j])
            if i != self.layers - 1:
                if sum(sum(self.node_active_encode[i+1])) == 0:
                    break
                
        last_features = [feature for feature in x_list if torch.is_tensor(feature)]
        last_features = [nn.Upsample(size=x.size()[2:], mode='bilinear')(feature) for feature in last_features]
        result = torch.cat(last_features, dim=1)
        result = self.classifier(result)
        
        return result
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        
        
if __name__ == '__main__':
    model = SuperNet(12, 4, 3, 3, 32, 32)
    model._initialize_weights()
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.size())