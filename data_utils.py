import os
from os import listdir
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToTensor, ToPILImage, CenterCrop, Resize
from torch.nn import init
import numpy

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def train_hr_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),  # convert a PIL image to tensor (H*W*C)
    ])

def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),  # convert a tensor to PIL image
        Resize(crop_size//upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

# cut image
def cut_image(image, count):
    width, height = image.size
    item_width = int(width / count)
    item_height = height
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, count):
        box = (i * item_width, 0, (i + 1) * item_width, item_height)
        box_list.append(box)
    image_list = [image.crop(box) for box in box_list]
    return image_list

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        image_list = cut_image(hr_image, 2)
        hr_image_i = image_list[0]
        hr_image_v = image_list[1]
        hr_image_i = self.hr_transform(hr_image_i)
        hr_image_v = self.hr_transform(hr_image_v)
        lr_image_i = self.lr_transform(hr_image_i)
        lr_image_v = self.lr_transform(hr_image_v)

        return lr_image_i, lr_image_v, hr_image_i, hr_image_v

    def __len__(self):
        return len(self.image_filenames)


class L1Loss(object):
    def __call__(self, input, target):
        return torch.abs(input - target).mean()

def save_ckpt(state, save_path='./log/x4', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)