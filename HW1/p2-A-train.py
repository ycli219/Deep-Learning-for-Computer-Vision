_exp_name = "model_A_vgg16fcn32" # ????

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
        img_transformed = self.transform(img)
        
        return img_transformed, self.labels[idx]
    
    def __len__(self):
        return len(self.files)

#=============================================================================
""" Model """

class vgg16fcn32s(nn.Module):

    def __init__(self, num_classes):
        super(vgg16fcn32s, self).__init__()

        self.num_classes = num_classes

        self.features = torchvision.models.vgg16(pretrained = True).features

        # Define the full convolutional layer of FCNs
        # The fc6 layer of the original VGG16 corresponds to a 7x7 convolutional layer with a 7x7 kernel size.
        self.fc6 = nn.Conv2d(512, 4096, kernel_size = 7, padding = 1)
        self.relu6 = nn.ReLU(inplace = True)

        # The fc7 layer of the original VGG16 corresponds to a convolutional layer with a kernel size of 1x1.
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size = 1, padding = 1)
        self.relu7 = nn.ReLU(inplace = True)

        # Convert the output to a category number, corresponding to the classification of each pixel.
        self.score_fr = nn.Conv2d(4096, 7, kernel_size = 1, padding = 1)

        self.relu = nn.ReLU(inplace = True)

        # Define the upsampling layer to upsample the feature image from the deeper layer back to the size of the input image.
        self.upscore = nn.ConvTranspose2d(7, 7, kernel_size = 64, stride = 32, bias = False)

    def forward(self, x):
        x = self.features(x)

        x = self.fc6(x)

        x = self.relu6(x)    

        x = self.fc7(x)

        x = self.relu7(x)    

        # Score Map with output as score per pixel for the corresponding category.
        x = self.score_fr(x)

        x = self.relu(x)

        # Up-sampling, up-sample the deep feature map directly back to the input map size.
        x = self.upscore(x)

        return x

#=============================================================================
"""Transforms"""

train_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

#=============================================================================
"""Configurations"""

writer = SummaryWriter()

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

num_class = 7

# Create a ResNet50 model
model = vgg16fcn32s(num_class)

model = model.to(device)

# The number of batch size.
batch_size = 8

# The number of training epochs.
n_epochs = 100

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

learning_rate = 0.003

#=============================================================================
""" Data loader"""

train_data = HW1P1Dataset("../hw1_data/p2_data/train", transform = train_transform)
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)

val_data = HW1P1Dataset("../hw1_data/p2_data/validation", transform = val_transform)
val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.01, epochs = n_epochs, steps_per_epoch = len(train_loader))

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

    outputs = torch.zeros((n_samples, height, width), dtype=torch.long)
    masks = torch.zeros((n_samples, height, width), dtype=torch.long)

    idx = 0

    for imgs, labels in tqdm(train_loader):

        imgs = imgs.to(torch.float32).to(device)
        labels = labels.to(torch.long).to(device)

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs)

        # Different spatial dimensions for logits and labels when dealing with semantic segmentation tasks 
        # (e.g. [8, 7, 320, 320] for logits and [8, 512, 512] for labels)
        # This will change the output of logits from [8, 7, 320, 320] to [8, 7, 512, 512].
        logits = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)
        
        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels)

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
    outputs = outputs.cpu().numpy()
    masks = masks.cpu().numpy()
    train_iou = mean_iou_score(outputs, masks)

    # Print the information.
    print(f"[ Train | {epoch:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, mIOU = {train_iou:.5f}")
    writer.add_scalar("Loss/Train", loss, epoch)

    torch.cuda.empty_cache()

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []

    n_samples = len(val_data)
    height, width = 512, 512

    outputs = torch.zeros((n_samples, height, width), dtype=torch.long)
    masks = torch.zeros((n_samples, height, width), dtype=torch.long)

    idx = 0

    # Iterate the validation set by batches.
    for imgs, labels in tqdm(val_loader):

        imgs = imgs.to(torch.float32).to(device)
        labels = labels.to(torch.long).to(device)

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs)

        logits = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)
        
        # We can still compute the loss (but not the gradient).
        #logits = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)
        loss = criterion(logits, labels)

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
    outputs = outputs.cpu().numpy()
    masks = masks.cpu().numpy()
    valid_iou = mean_iou_score(outputs, masks)

    # Print the information.
    print(f"[ Valid | {epoch:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, mIOU = {valid_iou:.5f}")
    writer.add_scalar("Loss/Validate", loss, epoch)

    torch.cuda.empty_cache()

    # save models
    if valid_iou > best_mIOU:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
        best_mIOU = valid_iou

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
