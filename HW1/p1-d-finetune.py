_exp_name = "model_D" # ????

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

from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn

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
train_list = glob.glob(os.path.join('../hw1_data/p1_data/office/train/', '*.jpg'))
val_list = glob.glob(os.path.join('../hw1_data/p1_data/office/val/', '*.jpg'))

#=============================================================================
"""Self-defined dataset"""

class HW1P1Dataset(Dataset):

    def __init__(self, file_list, transform = None):
        # Used to call the __init__ method of the parent class (i.e. Dataset) of HW1P1Dataset.
        # The second parameter of the super() function, self, is the instance object that ensures that the correct class parent is called.
        super(HW1P1Dataset, self).__init__()

        self.file_list = file_list
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        
        # ???? (Labeled or not).
        label = int(img_path.split("/")[-1].split("_")[0])
        return img_transformed, label
    
    def __len__(self):
        return len(self.file_list)

#=============================================================================
"""Transforms"""

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((128, 128)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

#=============================================================================
"""Configurations"""

writer = SummaryWriter()

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create a ResNet50 model
model = torchvision.models.resnet50(weights = None)

# Load pre-trained weights
model.load_state_dict(torch.load("../hw1_data/p1_data/pretrain_model_SL.pt")) # ????

""" ???? may need to Freeze the pre-trained layers """ 
for param in model.parameters():
    param.requires_grad = False 

# Getting the input features of the last layer of ResNet50
input_dim = model.fc.in_features

# Define FineTuneClassifier and replace the last layer of ResNet50 with it.
num_classes = 65

model.fc = torch.nn.Linear(input_dim, num_classes)

model = model.to(device)

# The number of batch size.
# The batch_size for fine-tuning can be set to a smaller value than for pre-training.
batch_size = 512

# The number of training epochs.
# The n_epochs for fine-tuning are usually less than the n_epochs for pre-training.
n_epochs = 150

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

#=============================================================================
""" Data loader"""

train_data = HW1P1Dataset(train_list, transform = train_transform)
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)

val_data = HW1P1Dataset(val_list, transform = val_transform)
val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.01, epochs = n_epochs, steps_per_epoch = len(train_loader))

#=============================================================================
""" Training """

# Initialize trackers, these are not parameters and should not be changed
best_acc = 0

for epoch in range(1, n_epochs + 1):
    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for imgs, labels in tqdm(train_loader):

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10)

        # Update the parameters with computed gradients.
        optimizer.step()

        scheduler.step()

        # Compute the accuracy for current batch.
        # logits.argmax(dim=-1) : This line of code is used to maximize the position (index) of the last dimension of logits (dim = -1)
        acc = (logits.argmax(dim = -1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)
    
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    writer.add_scalar("Loss/Train", loss, epoch)

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for imgs, labels in tqdm(val_loader):

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))
        
        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim = -1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
    
    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    writer.add_scalar("Loss/Validate", loss, epoch)

    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
        best_acc = valid_acc

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
    f.write(f'  * Best Accuracy : {best_acc}\n')