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

        color_optimizer = torch.optim.Adam(self.color_net.parameters(), lr=self.learning_rate)
        season_optimizer = torch.optim.Adam(self.season_net.parameters(), lr=self.learning_rate)
        part_optimizer = torch.optim.Adam(self.part_net.parameters(), lr=self.learning_rate)
        style_optimizer = torch.optim.Adam(self.style_net.parameters(), lr=self.learning_rate)
        category_optimizer = torch.optim.Adam(self.category_net.parameters(), lr=self.learning_rate)

        total_step = len(self.train_loader)

        color_scheduler = StepLR(color_optimizer, self.decay_epoch, gamma=0.5)
        season_scheduler = StepLR(season_optimizer, self.decay_epoch, gamma=0.5)
        part_scheduler = StepLR(season_optimizer, self.decay_epoch, gamma=0.5)
        style_scheduler = StepLR(style_optimizer, self.decay_epoch, gamma=0.5)
        category_scheduler = StepLR(category_optimizer, self.decay_epoch, gamma=0.5)

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

                outputs = self.color_net(images)
                loss = criterion(outputs, color)
                color_optimizer.zero_grad()
                loss.backward()
                color_optimizer.step()
                color_avg.update(loss.item())

                correct_color += outputs.argmax(dim=1).eq(color).sum().item()

                outputs = self.style_net(images)
                loss = criterion(outputs, style)
                style_optimizer.zero_grad()
                loss.backward()
                style_optimizer.step()
                style_avg.update(loss.item())

                correct_style += outputs.argmax(dim=1).eq(style).sum().item()

                outputs = self.part_net(images)
                loss = criterion(outputs, part)
                part_optimizer.zero_grad()
                loss.backward()
                part_optimizer.step()
                part_avg.update(loss.item())

                correct_part += outputs.argmax(dim=1).eq(part).sum().item()

                outputs = self.season_net(images)
                loss = criterion(outputs, season)
                season_optimizer.zero_grad()
                loss.backward()
                season_optimizer.step()
                season_avg.update(loss.item())

                correct_season += outputs.argmax(dim=1).eq(season).sum().item()

                outputs = self.category_net(images)
                loss = criterion(outputs, category)
                category_optimizer.zero_grad()
                loss.backward()
                category_optimizer.step()
                category_avg.update(loss.item())

                correct_category += outputs.argmax(dim=1).eq(category).sum().item()

                if step % 10 == 1:
                    print(f'Epoch [{epoch}/{self.num_epoch}], Step: [{step}/{total_step}], Color Loss: {color_avg.avg:.4f}, '
                          f'Season Loss: {season_avg.avg:.4f}, Part Loss: {part_avg.avg:.4f}, Style Loss: {style_avg.avg:.4f}, '
                          f'Category Loss: {category_avg.avg:.4f}')
                    print(f'Color: {correct_color/((step+1)*self.batch_size)*100:.4f}%, Style: {correct_style/((step+1)*self.batch_size)*100:.4f}%, '
                          f'Part: {correct_part/((step+1)*self.batch_size)*100:.4f}%, Season Category: {correct_season/((step+1)*self.batch_size)*100:.4f}%')

            torch.save(self.color_net.state_dict(), f'{self.checkpoint_dir}/{self.backbone}_color_checkpoint-{epoch}.pth')
            torch.save(self.season_net.state_dict(), f'{self.checkpoint_dir}/{self.backbone}_season_checkpoint-{epoch}.pth')
            torch.save(self.part_net.state_dict(), f'{self.checkpoint_dir}/{self.backbone}_part_checkpoint-{epoch}.pth')
            torch.save(self.style_net.state_dict(), f'{self.checkpoint_dir}/{self.backbone}_style_checkpoint-{epoch}.pth')
            torch.save(self.category_net.state_dict(), f'{self.checkpoint_dir}/{self.backbone}_category_checkpoint-{epoch}.pth')

            color_scheduler.step()
            season_scheduler.step()
            part_scheduler.step()
            style_scheduler.step()
            category_scheduler.step()

            update_vis_plot(epoch, color_avg.avg, color_plot, 'append')
            update_vis_plot(epoch, season_avg.avg, season_plot, 'append')
            update_vis_plot(epoch, part_avg.avg, part_plot, 'append')
            update_vis_plot(epoch, style_avg.avg, style_plot, 'append')
            update_vis_plot(epoch, category_avg.avg, category_plot, 'append')


    def build_model(self):
        self.color_net: nn.Module = Model(self.backbone, self.num_color).to(self.device)
        self.style_net: nn.Module = Model(self.backbone, self.num_style).to(self.device)
        self.part_net: nn.Module = Model(self.backbone, self.num_part).to(self.device)
        self.season_net: nn.Module = Model(self.backbone, self.num_season).to(self.device)
        self.category_net: nn.Module = Model(self.backbone, self.num_category).to(self.device)

        self.color_net.train()
        self.style_net.train()
        self.part_net.train()
        self.season_net.train()
        self.category_net.train()

        self.load_model()

    def load_model(self):
        print("[*] Load checkpoint in ", str(self.checkpoint_dir))

        model_parameter = glob(os.path.join(self.checkpoint_dir, f"{self.backbone}_color_checkpoint-*.pth"))

        if not model_parameter:
            print("[!] No checkpoint in ", str(self.checkpoint_dir))
            return

        epoch = []
        for s in model_parameter:
            epoch += [int(s.split('-')[-1].split('.')[0])]

        epoch.sort()

        self.epoch = epoch[-1]
        self.color_net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f"{self.backbone}_color_checkpoint-{self.epoch}.pth"), map_location=self.device))
        self.season_net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f"{self.backbone}_season_checkpoint-{self.epoch}.pth"), map_location=self.device))
        self.part_net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f"{self.backbone}_part_checkpoint-{self.epoch}.pth"), map_location=self.device))
        self.style_net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f"{self.backbone}_style_checkpoint-{self.epoch}.pth"), map_location=self.device))
        self.category_net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, f"{self.backbone}_category_checkpoint-{self.epoch}.pth"), map_location=self.device))

        print("[*] Load Model from %s: " % str(self.checkpoint_dir), epoch[-1])