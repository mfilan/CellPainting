import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, model, num_classes=9, train_net=False):
        super(CNN, self).__init__()
        self.model = model
        if not train_net:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        layer = self.model.conv1
        new_in_channels = 4
        new_layer = nn.Conv2d(in_channels=new_in_channels,
                              out_channels=layer.out_channels,
                              kernel_size=layer.kernel_size,
                              stride=layer.stride,
                              padding=layer.padding,
                              bias=layer.bias)
        copy_weights = 0
        new_layers_weight = new_layer.weight.clone()
        new_layers_weight[:, :layer.in_channels, :, :] = layer.weight.clone()
        for i in range(new_in_channels - layer.in_channels):
            channel = layer.in_channels + i
            new_layers_weight[:, channel:channel + 1, :, :] = layer.weight[:, copy_weights:copy_weights + 1, ::].clone()
        new_layer.weight = nn.Parameter(new_layers_weight)
        self.model.conv1 = new_layer

    def forward(self, images):
        features = self.model(images)
        return features