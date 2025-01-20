import os
import glob

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

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

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

batch_size = 512

train_list = glob.glob(os.path.join('../hw1_data/p1_data/office/train/', '*.jpg'))
train_data = HW1P1Dataset(train_list, transform = train_transform)
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory = True)

# 提取模型倒數第二層的輸出
def extract_features(feature_extractor, dataloader, device):
    feature_extractor.eval()  # 切換到評估模式
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = feature_extractor(inputs)
            outputs = torch.flatten(outputs, 1)  # 將輸出展平成為2D張量
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())
    return np.concatenate(features), np.concatenate(labels)

# t-SNE 降維
def apply_tsne(features):
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(features)

# 繪圖函數
def plot_tsne(features_2d, labels, title):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='jet', s=10)
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnet50(weights = None)

num_classes = 65

model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load("../p1-c-finetune/model_c_epoch1.ckpt"))

model = model.to(device)

# Use the submodel to extract the features of the second last layer
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# Extract the features of the epoch and perform t-SNE downsizing
features_epoch, labels_epoch = extract_features(feature_extractor, train_loader, device)
features_tsne_epoch = apply_tsne(features_epoch)

# Visualize the t-SNE result for the epoch
plot_tsne(features_tsne_epoch, labels_epoch, f't-SNE Visualization - Epoch1')


model2 = torchvision.models.resnet50(weights = None)

model2.fc = torch.nn.Linear(model2.fc.in_features, num_classes)

model2.load_state_dict(torch.load("../p1-c-finetune/model_c_epoch150.ckpt"))

model2 = model2.to(device)

# Use the submodel to extract the features of the second last layer
feature_extractor = torch.nn.Sequential(*list(model2.children())[:-1])

# Extract the features of the epoch and perform t-SNE downsizing
features_epoch, labels_epoch = extract_features(feature_extractor, train_loader, device)
features_tsne_epoch = apply_tsne(features_epoch)

# Visualize the t-SNE result for the epoch
plot_tsne(features_tsne_epoch, labels_epoch, f't-SNE Visualization - Epoch150')
