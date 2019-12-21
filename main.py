import os
from dataloader.dataloader import get_loader
from config.config import get_config
from src.train import Trainer

def main(config):
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    print('[*] Load Dataset')
    train_loader, test_loader = get_loader(config.dataset, config.image_size, config.batch_size)

    print('[*] Train')
    trainer = Trainer(config, train_loader)
    trainer.train()

if __name__ == "__main__":
    config = get_config()
    main(config)