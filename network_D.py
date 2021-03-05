import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hash_bit = 16
        self.base_model = getattr(torchvision.models, 'resnet18')(pretrained=True)
        self.conv1 = self.base_model.conv1
        self.bn1 = self.base_model.bn1
        self.relu = self.base_model.relu
        self.maxpool = self.base_model.maxpool
        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4
        self.avgpool = self.base_model.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.fc1 = nn.Linear(self.base_model.fc.in_features, self.base_model.fc.in_features)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(self.base_model.fc.in_features, self.base_model.fc.in_features)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(self.base_model.fc.in_features, self.hash_bit)
        self.last_layer = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.hash_layer = nn.Sequential(self.fc1, self.activation1, self.dropout, self.fc2, self.activation2, self.fc3,
                                        self.last_layer)

        self.iter_num = 0
        self.scale = 1

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)

        #y = self.last_layer(5*y)

        return y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = Net().to(device)

print(model)
summary(model, (3, 224, 224))
