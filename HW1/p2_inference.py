#=============================================================================

import numpy as np
import random
import torch

import os

import scipy.misc
import imageio.v2 as imageio

from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2

import torch.nn as nn
import torch.nn.functional as F

import sys

myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

#=============================================================================

test_transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#=============================================================================

def to_png(predict, path_name):
    predict = predict.detach().cpu().numpy()
    png_img = np.zeros((512, 512, 3), dtype = np.uint8)
    png_img[ np.where(predict == 0) ] = [0, 255, 255]
    png_img[ np.where(predict == 1) ] = [255, 255, 0]
    png_img[ np.where(predict == 2) ] = [255, 0, 255]
    png_img[ np.where(predict == 3) ] = [0, 255, 0]
    png_img[ np.where(predict == 4) ] = [0, 0, 255]
    png_img[ np.where(predict == 5) ] = [255, 255, 255]
    png_img[ np.where(predict == 6) ] = [0, 0, 0]
    imageio.imwrite(path_name, png_img)

#=============================================================================

class HW1P1Dataset(Dataset):

    def __init__(self, path, transform = None):
        super(HW1P1Dataset, self).__init__()

        self.path = path
        self.transform = transform
        self.files = sorted([x for x in os.listdir(path) if x.endswith(".jpg")])

        self.saved_png_names = sorted([x.split("_")[0] + '_mask.png' for x in self.files])
        
    def __getitem__(self, idx):
        img_name = self.files[idx]
        img = Image.open(os.path.join(self.path, img_name))
        img = torchvision.tv_tensors.Image(img) 

        img_transformed = self.transform(img)
        
        return img_transformed, self.saved_png_names[idx]
    
    def __len__(self):
        return len(self.files)

#=============================================================================

testing_images_directory = sys.argv[1]
output_images_directory = sys.argv[2]

device = "cuda" if torch.cuda.is_available() else "cpu"

num_class = 7

model = torchvision.models.segmentation.deeplabv3_resnet101(weights = 'DEFAULT')

model.classifier[4] = torch.nn.Conv2d(256, num_class, kernel_size=(1, 1))

model = model.to(device)

model.load_state_dict(torch.load("model_B_deeplabv3_resnet101_4_best.ckpt"))

#=============================================================================

model.eval()

batch_size = 4

test_data = HW1P1Dataset(testing_images_directory, transform = test_transform)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False, num_workers = 8, pin_memory = True)

for batch in test_loader:

    imgs, filenames = batch

    with torch.no_grad():
        logits = model(imgs.to(device))
    
    logits = F.interpolate(logits['out'], size=(512, 512), mode='bilinear', align_corners=False)

    logits = torch.argmax(logits, dim=1).cpu()

    result = list(zip(logits, filenames))

    for res in result:
        to_png(res[0], os.path.join(output_images_directory, res[1]))

