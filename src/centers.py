import os, configparser

from skimage.io import imread
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from model_layers.UNET import UNET, BatchNorm
from model_layers.RPN import RPN


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths
        self.model = UNET().eval()
        self.checkpoint = torch.load('ecseg.pt')
        self.model.load_state_dict(self.checkpoint)

    def __len__(self):
        return len(self.img_paths)

    def load_unet_output(self, img_path):
        img = imread(img_path)
        img = img.astype(np.float32)
        img /= img.max()
        img *= 255
        img = torch.Tensor(img).permute([2,0,1])[2:,...].unsqueeze(axis=0)
        with torch.no_grad():
            unet_output = self.model(img)
        return unet_output.squeeze(axis=0)

    def __getitem__(self, idx):
        return self.load_unet_output(self.img_paths[idx])

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    root_path = config.get('Centers', 'image_path')
    img_names = os.listdir(root_path)

    all_img_paths = [os.path.join(root_path, img) for img in img_names]
    dataset = ImageLoader(all_img_paths)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)

    output_path = os.path.join(root_path, 'centers')
    os.mkdir(output_path, exist_ok=True)

    NUM_ANCHORS = 10
    UNET_DIMS = 64
    rpn = RPN(NUM_ANCHORS, UNET_DIMS)
    rpn_weights = torch.load('RPN_weights.pt')
    rpn.load_state_dict(rpn_weights)

    for unet_output in tqdm(dataloader):
        rpn_output = rpn(unet_output)