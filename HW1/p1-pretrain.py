#=============================================================================
"""impoort packages"""

import numpy as np
import random
import torch

import os
import glob

from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torchvision
import torchvision.transforms as transforms

from byol_pytorch import BYOL

from tqdm.auto import tqdm

# The purpose of this code is to set random seeds to ensure the reproducibility of the model training process.
# When you conduct deep learning experiments, randomness (e.g. random partitioning of data, initialization of weights, etc.) may cause the results to vary from run to run.
# This code reduces this randomness by setting random seeds, thus making the experiment results more stable and reproducible.
myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

#=============================================================================
"""Save training data paths to list"""

# ../ means previous directory
train_list = glob.glob(os.path.join('../hw1_data/p1_data/mini/train/', '*.jpg')) 

#=============================================================================
"""Self-defined dataset"""

class HW1P1Dataset(Dataset):

    def __init__(self, file_list, transform = None):
        # Used to call the __init__ method of the parent class (i.e. Dataset) of HW1P1Dataset.
        # The second parameter of the super() function, self, is the instance object that ensures that the correct class parent is called.
        super(HW1P1Dataset, self).__init__() # ???? need self ????

        self.file_list = file_list
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        
        # may need to change (Labeled or not).
        label = -1
        return img_transformed, label
    
    def __len__(self):
        return len(self.file_list)

#=============================================================================
"""Transforms"""

# ???? data augmentations ????
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((128, 128)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

#=============================================================================
"""Configurations"""

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

resnet = torchvision.models.resnet50(weights = None) 

# BYOL model, and put it on the device specified.
learner = BYOL(
    resnet,
    image_size = 128,
    hidden_layer = 'avgpool'
).to(device)

opt = torch.optim.Adam(learner.parameters(), lr = 3e-4)

batch_size = 64
n_epochs = 300
save_every_n_epochs = 300

#=============================================================================
""" Data loader"""

train_data = HW1P1Dataset(train_list, transform = train_transform)
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 0, pin_memory = True)

#=============================================================================
""" Train """

for epoch in range(1, n_epochs + 1):

    print('running epoch: {}'.format(epoch))

    # Make sure the model is in train mode before training.
    learner.train()

    # Record information in training.
    train_loss = []

    for imgs, labels in tqdm(train_loader):

        # (Self-supervised) Directly output the loss, i.e., loss = model(x)
        loss = learner(imgs.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        opt.zero_grad()

        # Compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update).
        opt.step()

        # In a framework like BYOL, there are two networks:
        # 1. The online network: This network updates the weights based on the backpropagation of losses.
        # 2. The target network: the weights of this network are not updated directly by gradient descent, but are updated incrementally using the parameters of the online network.
        #    The update is done by using the moving average of the weights of the online network.
        learner.update_moving_average()

        # Record the loss.
        train_loss.append(loss.item())
    
    train_loss = sum(train_loss) / len(train_loss)

    # Print the information.
    print(f"[ Train | {epoch:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")

    if epoch % save_every_n_epochs == 0:
        torch.save(resnet.state_dict(), f'./ssl_bs{batch_size}_epoch{epoch}_nolabel.pt')




