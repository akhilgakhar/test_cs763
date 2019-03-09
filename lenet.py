import torch
import torch.nn as nn
from collections import OrderedDict

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([

            ('fire1', Fire(1, 3, 5, 5)),
            # ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('fire2', Fire(10, 3, 5, 5)),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('fire3', Fire(10, 3, 5, 5)),
            # ('s3', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('fire4', Fire(10, 3, 5, 5)),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('fire5', Fire(10, 3, 5, 5)),
            ('fire6', Fire(10, 3, 5, 5)),  
            ('s5', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('fire7', Fire(10, 3, 5, 5)),
            ('fire8', Fire(10, 3, 5, 5)),             
            ('a6', nn.AvgPool2d(4, stride=1)),
            # ('smax7', nn.LogSoftmax(dim=-1))



            # ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            # ('relu1', nn.ReLU()),
            # ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),

            # ('c3', Fire(6, 8, 8, 8)),

            # ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            # ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            # ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            # ('f6', nn.Linear(120, 84)),
            # ('relu6', nn.ReLU()),
            # ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output
