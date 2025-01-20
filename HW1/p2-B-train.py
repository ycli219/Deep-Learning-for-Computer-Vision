_exp_name = "model_B_deeplabv3_resnet101_4" # ????

#=============================================================================
"""impoort packages"""

import numpy as np
import random
import torch

import os
import glob

import scipy.misc
import imageio

from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torchvision
import torchvision.transforms as transforms

from torchvision.transforms import v2

from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.nn.functional as F

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

def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512))

    for i, file in enumerate(file_list):
        mask = imageio.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown 

    return masks

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f'%(i, iou))
    #print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

#=============================================================================
"""Self-defined dataset"""

class HW1P1Dataset(Dataset):

    def __init__(self, path, transform = None):
        # Used to call the __init__ method of the parent class (i.e. Dataset) of HW1P1Dataset.
        # The second parameter of the super() function, self, is the instance object that ensures that the correct class parent is called.
        super(HW1P1Dataset, self).__init__()

        self.path = path
        self.transform = transform
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])

        self.labels = read_masks(path)
        
    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path)
        img = torchvision.tv_tensors.Image(img) # ????

        ll = torchvision.tv_tensors.Mask(self.labels[idx])

        img_transformed, ll_transformed = self.transform(img, ll)
        
        return img_transformed, ll_transformed
    
    def __len__(self):
        return len(self.files)

#=============================================================================
""" Model """

#=============================================================================
"""Transforms"""

train_transform = v2.Compose([
    v2.RandomResizedCrop(size=(512, 512)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#=============================================================================
"""Configurations"""

writer = SummaryWriter()

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

num_class = 7

model = torchvision.models.segmentation.deeplabv3_resnet101(weights = 'DEFAULT')

model.classifier[4] = torch.nn.Conv2d(256, num_class, kernel_size=(1, 1))

model = model.to(device)

# The number of batch size.
batch_size = 4

# The number of training epochs.
n_epochs = 100

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

learning_rate = 0.003

#=============================================================================
""" Data loader"""

train_data = HW1P1Dataset("../hw1_data/p2_data/train", transform = train_transform)
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True)

val_data = HW1P1Dataset("../hw1_data/p2_data/validation", transform = val_transform)
val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True)

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.0005, epochs = n_epochs, steps_per_epoch = len(train_loader))

#=============================================================================
""" Training """

# Initialize trackers, these are not parameters and should not be changed
best_mIOU = 0

for epoch in range(1, n_epochs + 1):
    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []

    n_samples = len(train_data)
    height, width = 512, 512

    outputs = np.zeros((n_samples, height, width))
    masks = np.zeros((n_samples, height, width))

    idx = 0

    for imgs, labels in tqdm(train_loader):

        labels = labels.long()

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Different spatial dimensions for logits and labels when dealing with semantic segmentation tasks 
        # (e.g. [8, 7, 320, 320] for logits and [8, 512, 512] for labels)
        # This will change the output of logits from [8, 7, 320, 320] to [8, 7, 512, 512].
        logits = F.interpolate(logits['out'], size=(512, 512), mode='bilinear', align_corners=False)
        
        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Update the parameters with computed gradients.
        optimizer.step()

        #scheduler.step()

        logits = torch.argmax(logits, dim=1).cpu()  # [batch_size, 7, 512, 512] -> [batch_size, 512, 512]
        labels = labels.squeeze(1).cpu() # [batch_size, 1, 512, 512] -> [batch_size, 512, 512]

        b_size = logits.size(0)
        outputs[idx:idx + b_size, :, :] = logits
        masks[idx:idx + b_size, :, :] = labels

        idx += b_size

        # Record the loss and accuracy.
        train_loss.append(loss.item())
    
    train_loss = sum(train_loss) / len(train_loss)
    outputs = outputs
    masks = masks
    #train_iou = mean_iou_score(outputs, masks)

    # Print the information.
    #print(f"[ Train | {epoch:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, mIOU = {train_iou:.5f}")
    writer.add_scalar("Loss/Train", loss, epoch)

    torch.cuda.empty_cache()

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []

    n_samples = len(val_data)
    height, width = 512, 512

    outputs = np.zeros((n_samples, height, width))
    masks = np.zeros((n_samples, height, width))

    idx = 0

    # Iterate the validation set by batches.
    for imgs, labels in tqdm(val_loader):

        labels = labels.long()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        logits = F.interpolate(logits['out'], size=(512, 512), mode='bilinear', align_corners=False)
        
        # We can still compute the loss (but not the gradient).
        #logits = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)
        loss = criterion(logits, labels.to(device))

        logits = torch.argmax(logits, dim=1).cpu()  # [batch_size, 7, 512, 512] -> [batch_size, 512, 512]
        labels = labels.squeeze(1).cpu() # [batch_size, 1, 512, 512] -> [batch_size, 512, 512]

        b_size = logits.size(0)
        outputs[idx:idx + b_size, :, :] = logits
        masks[idx:idx + b_size, :, :] = labels

        idx += b_size

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
    
    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    outputs = outputs
    masks = masks
    valid_iou = mean_iou_score(outputs, masks)

    # Print the information.
    print(f"[ Valid | {epoch:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, mIOU = {valid_iou:.5f}")
    writer.add_scalar("Loss/Validate", loss, epoch)
    writer.add_scalar("Miou", valid_iou, epoch)

    torch.cuda.empty_cache()

    # save models
    if valid_iou > best_mIOU:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
        best_mIOU = valid_iou
    
    if epoch == 1 or epoch == n_epochs / 2 or epoch == n_epochs:
        torch.save(model.state_dict(), f"{_exp_name}_epoch{epoch}.ckpt")

writer.flush()
writer.close()

with open(f"./{_exp_name}_log.txt","a") as f:
    
    f.truncate(0)
    f.seek(0)

    f.write(f'<< {_exp_name} >>\n\n')
    f.write(f'Parameter Setting : \n\n')
    f.write(f'  * Model : \n')
    f.write(f'  {model}\n')   
    f.write(f'  * Batch Size : {batch_size}\n')
    f.write(f'  * N Epochs : {n_epochs}\n')
    f.write(f'Performance : \n\n')
    f.write(f'  * Best mIOU : {best_mIOU}\n')
