import torch.nn as nn
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.vgg import vgg11_bn


class Model(nn.Module):
    def __init__(self, backbone, num_classes):
        super(Model, self).__init__()
        if backbone == 'mobilenet_v2':
            base_network = mobilenet_v2(pretrained=True)
            self.features = base_network.features
            self.out_filters = base_network.last_channel
        elif backbone == 'vgg11_bn':
            base_network = vgg11_bn(pretrained=True)
            self.features = base_network.features
            self.out_filters = 512
        else:
            raise Exception(f"[!] Invalid model: {backbone}")

        self.avg_pool = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.out_filters, 512, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
