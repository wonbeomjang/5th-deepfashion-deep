import torch.nn as nn
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.vgg import vgg11_bn


class Model(nn.Module):
    def __init__(self, backbone, num_color, num_style, num_part, num_season, num_category):
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

        self.color_classifier = nn.Linear(512 * 7 * 7, num_color)
        self.style_classifier = nn.Linear(512 * 7 * 7, num_style)
        self.part_classifier = nn.Linear(512 * 7 * 7, num_part)
        self.season_classifier = nn.Linear(512 * 7 * 7, num_season)
        self.category_classifier = nn.Linear(512 * 7 * 7, num_category)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        color = self.color_classifier(x)
        style = self.style_classifier(x)
        part = self.part_classifier(x)
        season = self.season_classifier(x)
        category = self.category_classifier(x)

        return color, style, part, season, category
