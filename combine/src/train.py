import os
import torch
import torch.nn as nn
from glob import glob
from torch.optim.lr_scheduler import StepLR

from utils.utils import AverageMeter, create_vis_plot, update_vis_plot
from models.models import Model


class Trainer:
    def __init__(self, config, train_loader):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = config.checkpoint_dir
        self.learning_rate = config.lr
        self.train_loader = train_loader
        self.epoch = 0
        self.num_epoch = config.num_epoch
        self.backbone = config.backbone
        self.dataset = config.dataset
        self.decay_epoch = config.decay_epoch
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

    def train(self):
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        total_step = len(self.train_loader)
        scheduler = StepLR(optimizer, self.decay_epoch, gamma=0.5)

        color_avg = AverageMeter('color')
        season_avg = AverageMeter('season')
        part_avg = AverageMeter('part')
        style_avg = AverageMeter('style')
        category_avg = AverageMeter('category')

        color_plot = create_vis_plot('Epoch', 'Loss', 'Color')
        season_plot = create_vis_plot('Epoch', 'Loss', 'Season')
        part_plot = create_vis_plot('Epoch', 'Loss', 'Part')
        style_plot = create_vis_plot('Epoch', 'Loss', 'Style')
        category_plot = create_vis_plot('Epoch', 'Loss', 'Category')

        for epoch in range(self.epoch, self.num_epoch):
            color_avg.reset()
            season_avg.reset()
            part_avg.reset()
            style_avg.reset()
            category_avg.reset()

            correct_color = 0
            correct_style = 0
            correct_part = 0
            correct_season = 0
            correct_category = 0

            for step, (images, color, style, part, season, category) in enumerate(self.train_loader):
                images = images.to(self.device)
                color = color.to(self.device)
                style = style.to(self.device)
                part = part.to(self.device)
                season = season.to(self.device)
                category = category.to(self.device)

                color_prediction, style_prediction, part_prediction, season_prediction, category_prediction = self.net(images)

                color_loss = criterion(color_prediction, color)
                style_loss = criterion(style_prediction, style)
                part_loss = criterion(part_prediction, part)
                season_loss = criterion(season_prediction, season)
                category_loss = criterion(category_prediction, category)

                loss = color_loss + style_loss + part_loss + season_loss + category_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                color_avg.update(color_loss.item())
                style_avg.update(style_loss.item())
                part_avg.update(part_loss.item())
                season_avg.update(season_loss.item())
                category_avg.update(category_loss.item())

                correct_color += color_prediction.argmax(dim=1).eq(color).sum().item()
                correct_style += style_prediction.argmax(dim=1).eq(style).sum().item()
                correct_part += part_prediction.argmax(dim=1).eq(part).sum().item()
                correct_season += season_prediction.argmax(dim=1).eq(season).sum().item()
                correct_category += category_prediction.argmax(dim=1).eq(category).sum().item()

                if step % 10 == 1:
                    print(f'Epoch [{epoch}/{self.num_epoch}], Step: [{step}/{total_step}], Color Loss: {color_avg.avg:.4f}, '
                          f'Season Loss: {season_avg.avg:.4f}, Part Loss: {part_avg.avg:.4f}, Style Loss: {style_avg.avg:.4f}, '
                          f'Category Loss: {category_avg.avg:.4f}')
                    print(f'Color: {correct_color/((step+1)*self.batch_size)*100:.4f}%, Style: {correct_style/((step+1)*self.batch_size)*100:.4f}%, '
                          f'Part: {correct_part/((step+1)*self.batch_size)*100:.4f}%, Season Category: {correct_season/((step+1)*self.batch_size)*100:.4f}%')

            torch.save(self.net.state_dict(), f'{self.checkpoint_dir}/f"{self.backbone}_checkpoint-{epoch}.pth')

            scheduler.step()

            update_vis_plot(epoch, color_avg.avg, color_plot, 'append')
            update_vis_plot(epoch, season_avg.avg, season_plot, 'append')
            update_vis_plot(epoch, part_avg.avg, part_plot, 'append')
            update_vis_plot(epoch, style_avg.avg, style_plot, 'append')
            update_vis_plot(epoch, category_avg.avg, category_plot, 'append')

    def build_model(self):
        self.net: nn.Module = Model(self.backbone, self.num_color, self.num_style, self.num_part, self.num_season,
                                    self.num_category).to(self.device)

        self.load_model()

    def load_model(self):
        print("[*] Load checkpoint in ", str(self.checkpoint_dir))

        if not os.listdir(self.checkpoint_dir):
            print("[!] No checkpoint in ", str(self.checkpoint_dir))
            return

        model_parameter = glob(os.path.join(self.checkpoint_dir, f"{self.backbone}_checkpoint-{self.epoch}-*.pth"))

        epoch = []
        for s in model_parameter:
            epoch += [int(s.split('-')[-1].split('.')[0])]

        epoch.sort()
        self.epoch = epoch[-1]
        self.net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f"{self.backbone}_checkpoint-{self.epoch}.pth"), map_location=self.device))

        print("[*] Load Model from %s: " % str(self.checkpoint_dir), epoch[-1])