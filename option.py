import argparse

parser = argparse.ArgumentParser(description='HKDnet')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=0,
                    help='number of threads for data loading')
parser.add_argument('--cpu', default=False,
                    help='use cpu only')
parser.add_argument('--cuda', default=True, help='use cuda')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--crop_size', type=int, default=144)

parser.add_argument('--n_colors', default=1,
                    help='n_colors')
parser.add_argument('--data_train_hr', type=str, default=r'./Datasets/images',
                    help='train dataset name')
parser.add_argument('--data_val_hr', type=str, default=r'./Valset',
                    help='Val dataset name')
parser.add_argument('--scale', default=2,
                    help='super resolution scale')

# Model specifications

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--n_resblocks', type=int, default=20,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')

# Training specifications
parser.add_argument('--epochs', type=int, default=250,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=20,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='SGD',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Log specifications
parser.add_argument('--log_interval', type=int, default='100',
                    help='number of images after which the training loss is logged')

# options for residual group and feature channel reduction
parser.add_argument('--n_resgroups', type=int, default=5,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')


args = parser.parse_args()



