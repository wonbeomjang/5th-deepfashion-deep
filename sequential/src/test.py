import os
import torch
import torch.nn as nn
from glob import glob

from utils.utils import create_vis_plot, update_vis_plot
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

        color_plot = create_vis_plot('Epoch', 'Accuracy', 'Color')
        season_plot = create_vis_plot('Epoch', 'Accuracy', 'Season')
        part_plot = create_vis_plot('Epoch', 'Accuracy', 'Part')
        style_plot = create_vis_plot('Epoch', 'Accuracy', 'Style')
        category_plot = create_vis_plot('Epoch', 'Accuracy', 'Category')

        with torch.no_grad():
            for epoch in range(self.epoch + 1):

                correct_color = 0
                correct_style = 0
                correct_part = 0
                correct_season = 0
                correct_category = 0

                self.color_net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f"{self.backbone}_color_checkpoint-{epoch}.pth"), map_location=self.device))
                self.season_net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f"{self.backbone}_season_checkpoint-{epoch}.pth"), map_location=self.device))
                self.part_net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f"{self.backbone}_part_checkpoint-{epoch}.pth"), map_location=self.device))
                self.style_net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f"{self.backbone}_style_checkpoint-{epoch}.pth"), map_location=self.device))
                self.category_net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f"{self.backbone}_category_checkpoint-{epoch}.pth"), map_location=self.device))

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

                print(f'Epoch: {epoch}, Color: {correct_color / len(self.test_loader) / self.batch_size * 100:.4f}%, Style: {correct_style / len(self.test_loader) / self.batch_size * 100:.4f}%, '
                      f'Part: {correct_part / len(self.test_loader) / self.batch_size * 100:.4f}%, Season: {correct_season / len(self.test_loader) / self.batch_size * 100:.4f}% '
                      f'Category: {correct_category / len(self.test_loader) / self.batch_size * 100:.4f}%')

                update_vis_plot(epoch, correct_color / len(self.test_loader) / self.batch_size * 100, color_plot, 'append')
                update_vis_plot(epoch, correct_style / len(self.test_loader) / self.batch_size * 100, style_plot, 'append')
                update_vis_plot(epoch, correct_part / len(self.test_loader) / self.batch_size * 100, part_plot, 'append')
                update_vis_plot(epoch, correct_season / len(self.test_loader) / self.batch_size * 100, season_plot, 'append')
                update_vis_plot(epoch, correct_category / len(self.test_loader) / self.batch_size * 100, category_plot, 'append')

    def build_model(self):
            self.color_net: nn.Module = Model(self.backbone, self.num_color).to(self.device)
            self.style_net: nn.Module = Model(self.backbone, self.num_style).to(self.device)
            self.part_net: nn.Module = Model(self.backbone, self.num_part).to(self.device)
            self.season_net: nn.Module = Model(self.backbone, self.num_season).to(self.device)
            self.category_net: nn.Module = Model(self.backbone, self.num_category).to(self.device)

            self.color_net.eval()
            self.style_net.eval()
            self.part_net.eval()
            self.season_net.eval()
            self.category_net.eval()

            self.load_model()

    def load_model(self):
        print("[*] Load checkpoint in ", str(self.checkpoint_dir))

        model_parameter = glob(os.path.join(self.checkpoint_dir, f"{self.backbone}_color_checkpoint-*.pth"))

        if not model_parameter:
            raise Exception("[!] No checkpoint in ", str(os.path.join(self.checkpoint_dir, f"{self.backbone}_color_checkpoint-*.pth")))

        epoch = []
        for s in model_parameter:
            epoch += [int(s.split('-')[-1].split('.')[0])]

        epoch.sort()

        self.epoch = epoch[-1]
