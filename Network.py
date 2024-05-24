## YOUR CODE HERE
import torch
import tensorflow as tf
import torch.nn as nn
import math
from torch.functional import Tensor
import torch.nn.functional as F

"""This script defines the network.
"""

class MyNetwork(nn.Module):

    def __init__(self, config):
        super(MyNetwork, self).__init__()

        depth = config["depth"]
        num_classes = config["num_classes"]
        # no of groups in each block
        cardinality = config["cardinality"]  
        dropRate = config["dropRate"]

        num_channels = [16, 16 * cardinality, 32 * cardinality, 64 * cardinality]
        self.start_layer = nn.Conv2d(3, num_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.stack_layers= nn.ModuleList()
        for i in range(3):
            self.stack_layers.append(stack_resnet_layers(depth,num_channels[i],num_channels[i+1],resnetblock,1 if i == 0 else 2,dropRate))
        self.bn_relu = nn.Sequential(nn.BatchNorm2d(num_channels[3], eps= 1e-5, momentum = 0.995), nn.ReLU())
        self.output_layer = output_layer(num_channels[3],num_classes)
        self.build_network()
    
    def build_network(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                params = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / params))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.bias.data.zero_()

    def forward(self, inputs):
        '''
    	Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Used by operations that work differently
                in training and testing phases such as batch normalization.
        Return:
            The output Tensor of the network.
    	'''
        outputs = self.start_layer(inputs)
        for i in range(3):
            outputs = self.stack_layers[i](outputs)
        outputs = self.bn_relu(outputs)
        outputs = self.output_layer(outputs)
        return outputs
    
# stack layers
class stack_resnet_layers(nn.Module):
    def __init__(self, num_layers, kernels_in, kernels_out, block, stride, dropRate=0.0):
        super(stack_resnet_layers, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(block(kernels_in, kernels_out, stride, dropRate))
            else:
                self.layers.append(block(kernels_out, kernels_out, 1, dropRate))

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output

# resnet block with multiple pathways
class resnetblock(nn.Module):
    def __init__(self, filters_in, filters_out, strides, dropRate=0.0, num_pathways=2) -> None:
        super(resnetblock, self).__init__()
        self.num_pathways = num_pathways
        
        self.bnorms = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.convs = nn.ModuleList()

        for _ in range(num_pathways):
            self.bnorms.append(nn.BatchNorm2d(filters_in))
            self.relus.append(nn.ReLU())
            self.convs.append(nn.Conv2d(filters_in, filters_out, kernel_size=3, stride=strides, bias=False, padding=1))

        self.bnorm2 = nn.BatchNorm2d(num_pathways * filters_out)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropRate)
        self.conv2 = nn.Conv2d(num_pathways * filters_out, filters_out, kernel_size=3, stride=1, bias=False, padding=1)
        
        self.flag = True
        if not (filters_in == filters_out):
            self.flag = False
            self.bnorm_shortcut = nn.BatchNorm2d(filters_in)
            self.relu_shortcut = nn.ReLU()
            self.conv_shortcut = nn.Conv2d(filters_in, filters_out, kernel_size=1, stride=strides, bias=False, padding=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i in range(self.num_pathways):
            output = self.convs[i](self.relus[i](self.bnorms[i](inputs)))
            outputs.append(output)
        output = torch.cat(outputs, dim=1)
        output = self.conv2(self.dropout(self.relu2(self.bnorm2(output))))
        if self.flag:
            output += inputs
        else:
            shortcut = self.conv_shortcut(self.relu_shortcut(self.bnorm_shortcut(inputs)))
            output += shortcut
        
        return output

class output_layer(nn.Module):
    def __init__(self, filters, num_classes) -> None:
        super(output_layer, self).__init__()
        self.classes = nn.Linear(filters, num_classes)
        self.filters = filters
    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs
        output = F.avg_pool2d(output,8)
        output = output.view(-1, self.filters)
        output = self.classes(output)
        return output

