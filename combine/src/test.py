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

                self.net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f"{self.backbone}_checkpoint-{self.epoch}.pth"), map_location=self.device))

                for step, (images, color, style, part, season, category) in enumerate(self.test_loader):
                    images = images.to(self.device)
                    color = color.to(self.device)
                    style = style.to(self.device)
                    part = part.to(self.device)
                    season = season.to(self.device)
                    category = category.to(self.device)

                    color_prediction, style_prediction, part_prediction, season_prediction, category_prediction = self.net(images)

                    correct_color += color_prediction.argmax(dim=1).eq(color).sum().item()
                    correct_style += style_prediction.argmax(dim=1).eq(style).sum().item()
                    correct_part += part_prediction.argmax(dim=1).eq(part).sum().item()
                    correct_season += season_prediction.argmax(dim=1).eq(season).sum().item()
                    correct_category += category_prediction.argmax(dim=1).eq(category).sum().item()

                print(f'Epoch: {epoch}, Color: {correct_color / len(self.test_loader) / self.batch_size * 100:.4f}%, Style: {correct_style / len(self.test_loader) / self.batch_size * 100:.4f}%, '
                      f'Part: {correct_part / len(self.test_loader) / self.batch_size * 100:.4f}%, Season: {correct_season / len(self.test_loader) / self.batch_size * 100:.4f}% '
                      f'Category: {correct_category / len(self.test_loader) / self.batch_size * 100:.4f}%')

                update_vis_plot(epoch, correct_color / len(self.test_loader) / self.batch_size * 100, color_plot, 'append')
                update_vis_plot(epoch, correct_style / len(self.test_loader) / self.batch_size * 100, style_plot, 'append')
                update_vis_plot(epoch, correct_part / len(self.test_loader) / self.batch_size * 100, part_plot, 'append')
                update_vis_plot(epoch, correct_season / len(self.test_loader) / self.batch_size * 100, season_plot, 'append')
                update_vis_plot(epoch, correct_category / len(self.test_loader) / self.batch_size * 100, category_plot, 'append')

    def build_model(self):
        self.net: nn.Module = Model(self.backbone, self.num_color, self.num_style, self.num_part, self.num_season,
                                    self.num_category).to(self.device)

        self.load_model()

    def load_model(self):
        print("[*] Load checkpoint in ", str(self.checkpoint_dir))

        model_parameter = glob(os.path.join(self.checkpoint_dir, f"{self.backbone}_checkpoint-{self.epoch}-*.pth"))

        if not model_parameter:
            raise Exception("[!] No checkpoint in ", str(self.checkpoint_dir))

        epoch = []
        for s in model_parameter:
            epoch += [int(s.split('-')[-1].split('.')[0])]

        epoch.sort()

        self.epoch = epoch[-1]
