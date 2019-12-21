import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--num_classes', type=int, default=2, help='number of model output channels')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_epoch', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--checkpoint_dir', default='checkpoints', help="path to saved models (to continue training)")
parser.add_argument('--dataset', default='dataset', help="")
parser.add_argument('--backbone', default='mobilenet_v2', help="backbone network")
parser.add_argument('--decay_epoch', type=int, default=50, help="decay step")

parser.add_argument('--num_color', type=int,  default=19, help="the number of color")
parser.add_argument('--num_style', type=int,  default=14, help="the number of style")
parser.add_argument('--num_part', type=int,  default=6, help="the number of season")
parser.add_argument('--num_season', type=int,  default=6, help="the number of season")
parser.add_argument('--num_category', type=int,  default=40, help="the number of category")

def get_config():
    return parser.parse_args()