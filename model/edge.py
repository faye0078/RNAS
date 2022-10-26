import torch
from torch import nn
BatchNorm2d = torch.nn.SyncBatchNorm

class BasicEdge(nn.Module):
    
    def __init__(self, input_channels, output_channels):
        """Basic Edge in supernet

        Args:
            input_channels (int): input channel of edge
            output_channels (int): output channel of edge

        Raises:
            ValueError: input_channels must be equal to output_channels
        """
        if input_channels != output_channels:
            raise ValueError(f'input_channels != output_channels: {input_channels} != {output_channels}, basic edge only support same input and output channels')
        super(BasicEdge, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        output = x + residual
        output = self.relu(output)
        
        return output
    
class KeepEdge(nn.Module):
    
    def __init__(self, input_channels, output_channels):
        """_summary_

        Args:
            input_channels (int): input channel of edge
            output_channels (int): output channel of edge

        Raises:
            ValueError: input_channels must be equal to output_channels
        """
        if input_channels != output_channels:
            raise ValueError(f'input_channels != output_channels: {input_channels} != {output_channels}, keep edge only support same input and output channels')
        super(KeepEdge, self).__init__()
        keep = []
        keep.append(nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=1, bias=False))
        keep.append(BatchNorm2d(output_channels))
        
        self.keep = nn.Sequential(*keep)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        
        x = self.keep(x)
        x = self.relu(x)
        
        return x

class DownsampleEdge(nn.Module):
    
    def __init__(self, input_channels, output_channels, down_num):
        """_summary_

        Args:
            input_channels (int): input channel of edge
            output_channels (int): output channel of edge
            down_num (int): downsample times

        Raises:
            ValueError: input_channels must be less than output_channels
        """        
        if input_channels >= output_channels:
            raise ValueError(f'input_channels >= output_channels: {input_channels} >= {output_channels}, downsample node only support input_channels < output_channels')
        super(DownsampleEdge, self).__init__()
        downsample = []
        for i in range(down_num):
            _input_channels = input_channels if i == 0 else _output_channels
            _output_channels = output_channels // (2 ** (i + 1))
            downsample.append(nn.Conv2d(_input_channels, _output_channels, kernel_size=3, stride=2, padding=1, bias=False))
            downsample.append(BatchNorm2d(_output_channels))
        
        self.downsample = nn.Sequential(*downsample)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.downsample(x)
        return self.relu(x)
    
class UpsampleEdge(nn.Module):
    
    def __init__(self, input_channels, output_channels, up_num):
        """_summary_

        Args:
            input_channels (int): input channel of edge
            output_channels (int): output channel of edge
            up_num (int): upsample times 

        Raises:
            ValueError: input_channels must be greater than output_channels
        """        
        if input_channels <= output_channels:
            raise ValueError(f'input_channels <= output_channels: {input_channels} <= {output_channels}, upsample node only support input_channels > output_channels')
        super(UpsampleEdge, self).__init__()
        upsample = []
        for i in range(up_num):
            _input_channels = input_channels if i == 0 else _output_channels
            _output_channels = output_channels * (2 ** (i + 1))
            upsample.append(nn.ConvTranspose2d(_input_channels, _output_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False))
            upsample.append(BatchNorm2d(_output_channels))
        
        self.upsample = nn.Sequential(*upsample)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.upsample(x)
        return self.relu(x)
        