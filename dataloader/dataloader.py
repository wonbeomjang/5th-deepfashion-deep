import torch.utils.data
import os
import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import random_split


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size):
        self.data_dir = data_dir
        self.image_size = image_size

        if not os.path.exists(self.data_dir):
            raise Exception(" [!] {}  not exists.".format(self.data_dir))

        self.color_label = self._make_label('color')
        self.style_label = self._make_label('style')
        self.part_label = self._make_label('part')
        self.season_label = self._make_label('season')
        self.category_label = self._make_label('category')

        label_file = open(os.path.join(self.data_dir, 'labels', 'labels.csv'))
        lines = label_file.readlines()
        self.data = [line[:-1] for line in lines]
        self.data = self.data[1:]

    def __getitem__(self, item):
        data = self.data[item]

        image_file_name, color, style, part, season, category = data.split(',')

        image = Image.open(os.path.join(self.data_dir, 'images', image_file_name)).convert('RGB')

        transform = transforms.Compose([
            transforms.CenterCrop(min(image.size[0], image.size[1])),
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        return [transform(image), self.color_label[color], self.style_label[style], self.part_label[part], self.season_label[season], self.category_label[category]]

    def __len__(self):
        return len(self.data)

    def _make_label(self, label_name):
        labels = {}
        file = open(os.path.join(self.data_dir, 'labels', f'{label_name}.csv'), 'r')
        line = file.readline()

        for i, label, in enumerate(line.split(',')):
            labels[label] = i

        return labels


def get_loader(data_dir, image_size, batch_size):
    dataset = Dataset(data_dir, image_size)

    train_length = int(0.9 * len(dataset))
    test_length = len(dataset) - train_length

    train_dataset, test_dataset = random_split(dataset, (train_length, test_length))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_loader, test_loader
