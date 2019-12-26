import os
import torch
import torch.nn as nn
from glob import glob
from torch.optim.lr_scheduler import StepLR

from utils.utils import AverageMeter, create_vis_plot, update_vis_plot
from models.models import Model


class Tester:
    def __init__(self, config, test_loader):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = config.checkpoint_dir
        self.test_loader = test_loader
        self.backbone = config.backbone
        self.dataset = config.dataset
        self.batch_size = config.batch_size

        label_data_file = open(os.path.join(self.dataset, 'label_data.txt'))
        self.num_color, self.num_style, self.num_part, self.num_season, self.num_category = label_data_file.readlines()[1].split(',')
        self.num_color = int(self.num_color)
        self.num_style = int(self.num_style)
        self.num_part = int(self.num_part)
        self.num_season = int(self.num_season)
        self.num_category = int(self.num_category)
        label_data_file.close()

        self.build_model()

    def test(self):
        correct_color = 0
        correct_style = 0
        correct_part = 0
        correct_season = 0
        correct_category = 0

        for step, (images, color, style, part, season, category) in enumerate(self.test_loader):
            images = images.to(self.device)
            color = color.to(self.device)
            style = style.to(self.device)
            part = part.to(self.device)
            season = season.to(self.device)
            category = category.to(self.device)

            outputs = self.color_net(images)
            correct_color += outputs.argmax(dim=1).eq(color).sum().item()

            outputs = self.style_net(images)
            correct_style += outputs.argmax(dim=1).eq(style).sum().item()

            outputs = self.part_net(images)
            correct_part += outputs.argmax(dim=1).eq(part).sum().item()

            outputs = self.season_net(images)
            correct_season += outputs.argmax(dim=1).eq(season).sum().item()

            outputs = self.category_net(images)
            correct_category += outputs.argmax(dim=1).eq(category).sum().item()

            if step % 100 == 1:
                print(f'Color: {correct_color / (step * self.batch_size) * 100:.4f}%, Style: {correct_style / (step * self.batch_size) * 100:.4f}%, '
                      f'Part: {correct_part / (step * self.batch_size) * 100:.4f}%, Season Category: {correct_season / (step * self.batch_size) * 100:.4f}')

    def build_model(self):
            self.color_net: nn.Module = Model(self.backbone, self.num_color).to(self.device)
            self.style_net: nn.Module = Model(self.backbone, self.num_style).to(self.device)
            self.part_net: nn.Module = Model(self.backbone, self.num_part).to(self.device)
            self.season_net: nn.Module = Model(self.backbone, self.num_season).to(self.device)
            self.category_net: nn.Module = Model(self.backbone, self.num_category).to(self.device)

            self.color_net.to(self.device)
            self.style_net.to(self.device)
            self.part_net.to(self.device)
            self.season_net.to(self.device)
            self.category_net.to(self.device)

            self.load_model()

    def load_model(self):
        print("[*] Load checkpoint in ", str(self.checkpoint_dir))

        if not os.listdir(self.checkpoint_dir):
            print("[!] No checkpoint in ", str(self.checkpoint_dir))
            return

        model_parameter = glob(os.path.join(self.checkpoint_dir, "color_checkpoint-*.pth"))

        epoch = []
        for s in model_parameter:
            epoch += [int(s.split('-')[-1].split('.')[0])]

        epoch.sort()

        self.epoch = epoch[-1]
        self.color_net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f"color_checkpoint-{self.epoch}.pth"), map_location=self.device))
        self.season_net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f"season_checkpoint-{self.epoch}.pth"), map_location=self.device))
        self.part_net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f"part_checkpoint-{self.epoch}.pth"), map_location=self.device))
        self.style_net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f"style_checkpoint-{self.epoch}.pth"), map_location=self.device))
        self.category_net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f"category_checkpoint-{self.epoch}.pth"), map_location=self.device))

        print("[*] Load Model from %s: " % str(self.checkpoint_dir), epoch[-1])