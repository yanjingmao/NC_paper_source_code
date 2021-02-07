import torch
import torch.nn as nn
import dlutil as dl


class SeparableConv2D(nn.Module):
    '''
    Definition of Separable Convolution.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, depth_multiplier=1, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros'):
        super(SeparableConv2D, self).__init__()
        depthwise_conv_out_channels = in_channels * depth_multiplier
        self.depthwise_conv = nn.Conv2d(in_channels, depthwise_conv_out_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias, padding_mode=padding_mode)
        self.pointwise_conv = nn.Conv2d(depthwise_conv_out_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.depthwise_conv(x)
        output = self.pointwise_conv(x)
        return output


class Block1(nn.Module):
    '''
    Definition of Block 1.
    '''
    def __init__(self, in_channels):
        super(Block1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, (3, 3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.out_channels = 64

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        return x

class Block2(nn.Module):
    '''
    Definition of Block 2.
    '''
    def __init__(self, in_channels):
        super(Block2, self).__init__()
        self.r_conv1 = nn.Conv2d(in_channels, 128, (1, 1), stride=(2, 2), bias=False)
        self.r_bn1 = nn.BatchNorm2d(128)

        self.conv1 = SeparableConv2D(in_channels, 128, (3, 3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = SeparableConv2D(128, 128, (3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.mp3 = nn.MaxPool2d((3, 3), stride=(2, 2), padding=1)
        self.out_channels = 128

    def forward(self, x):
        # Shortcut
        rx = self.r_conv1(x)
        rx = self.r_bn1(rx)
        # Main way
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.mp3(x)
        # Confluence
        x = x + rx
        return x


class Block3(nn.Module):
    '''
    Definition of Block 3.
    '''
    def __init__(self, in_channels):
        super(Block3, self).__init__()
        self.r_conv1 = nn.Conv2d(in_channels, 256, (1, 1), stride=(2, 2), bias=False)
        self.r_bn1 = nn.BatchNorm2d(256)

        self.conv1 = SeparableConv2D(in_channels, 256, (3, 3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = SeparableConv2D(256, 256, (3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.mp3 = nn.MaxPool2d((3, 3), stride=(2, 2), padding=1)
        self.out_channels = 256

    def forward(self, x):
        # Shortcut
        rx = self.r_conv1(x)
        rx = self.r_bn1(rx)
        # Main way
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.mp3(x)
        # Confluence
        x = x + rx
        return x

class Block4(nn.Module):
    '''
    Definition of Block 4.
    '''
    def __init__(self, in_channels):
        super(Block4, self).__init__()
        self.conv1 = SeparableConv2D(in_channels, 256, (3, 3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = SeparableConv2D(256, 256, (3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = SeparableConv2D(256, 256, (3, 3), padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.out_channels = 256

    def forward(self, x):
        # Shortcut
        rx = x
        # Main way
        x = torch.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        # Confluence
        x = x + rx
        return x

class Block5(nn.Module):
    '''
    Definition of Block 5.
    '''
    def __init__(self, in_channels):
        super(Block5, self).__init__()
        self.r_conv1 = nn.Conv2d(in_channels, 512, (1, 1), stride=(2, 2), bias=False)
        self.r_bn1 = nn.BatchNorm2d(512)

        self.conv1 = SeparableConv2D(in_channels, 256, (3, 3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = SeparableConv2D(256, 512, (3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.mp3 = nn.MaxPool2d((3, 3), stride=(2, 2), padding=1)
        self.out_channels = 512

    def forward(self, x):
        # Shortcut
        rx = self.r_conv1(x)
        rx = self.r_bn1(rx)
        # Main way
        x = torch.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.mp3(x)
        # Confluence
        x = x + rx
        return x

class Block6(nn.Module):
    '''
    Definition of Block 6.
    '''
    def __init__(self, in_channels):
        super(Block6, self).__init__()
        self.conv1 = SeparableConv2D(in_channels, 1024, (3, 3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = SeparableConv2D(1024, 2048, (3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(2048)
        self.out_channels = 2048
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        return x

class Network(nn.Module):
    '''
    Definition of the whole network with Block[1-6] utilized.
    '''
    def __init__(self, in_channels, num_classes, num_middle_layers=4):
        super(Network, self).__init__()
        self.block1 = Block1(in_channels)
        self.block2 = Block2(self.block1.out_channels)
        self.block3 = Block3(self.block2.out_channels)
        assert num_middle_layers >= 0, f'Invalid number of layers, {num_middle_layers}'
        if num_middle_layers != 0:
            self.block4_lst = nn.ModuleList([Block4(self.block3.out_channels) for _ in range(num_middle_layers)])
            self.block5 = Block5(self.block4_lst[0].out_channels)
        else:
            self.block5 = Block5(self.block3.out_channels)
        self.block6 = Block6(self.block5.out_channels)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.final = nn.Linear(self.block6.out_channels, num_classes)
        

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x) # half-sized length and high
        x = self.block3(x) # half-sized length and high
        for i in range(len(self.block4_lst)):
            x = self.block4_lst[i](x)
        x = self.block5(x) # half-sized length and high
        x = self.block6(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.final(x)
        return x
        