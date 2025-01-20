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

import sys
import csv
import pandas as pd

myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

#=============================================================================

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

#=============================================================================

class HW1P1Dataset(Dataset):

    def __init__(self, file_list, transform = None):
        super(HW1P1Dataset, self).__init__()

        self.file_list = file_list
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        
        fname = img_path.split("/")[-1]
        return img_transformed, fname
    
    def __len__(self):
        return len(self.file_list)

#=============================================================================

path_of_the_images_csv_file = sys.argv[1]
path_of_the_folder_containing_images = sys.argv[2]
path_of_output_csv_file = sys.argv[3]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torchvision.models.resnet50(weights = None)

input_dim = model.fc.in_features

num_classes = 65

model.fc = torch.nn.Linear(input_dim, num_classes)

model = model.to(device)

model.load_state_dict(torch.load("model_c_best.ckpt"))

#=============================================================================

batch_size = 64

test_img_info = pd.read_csv(path_of_the_images_csv_file)

filename_info = test_img_info['filename'].tolist()

for i in range(0, len(filename_info)):
    tmp = os.path.join(path_of_the_folder_containing_images, filename_info[i])
    filename_info[i] = tmp

test_data = HW1P1Dataset(filename_info, transform = test_transform)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)

file_name = []
pre_label = []

model.eval()

for batch in test_loader:

    imgs, filenames = batch

    with torch.no_grad():
        logits = model(imgs.to(device))

    file_name.extend(filenames)

    pre_label.extend(logits.argmax(dim=-1).flatten().detach().tolist())

#=============================================================================

filename_and_label = list(zip(file_name, pre_label))

new_filename_info = test_img_info['filename'].tolist()

for res in filename_and_label:

    idx = new_filename_info.index(res[0])

    test_img_info['label'][idx] = res[1]

test_img_info.to_csv(path_of_output_csv_file, index = False) 








