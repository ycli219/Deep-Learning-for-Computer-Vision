import torch
from torch import nn
import numpy as np
import random
import os
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)

            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()

        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()

        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_digit_classes=10, n_dataset_classes=2):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_digit_classes = n_digit_classes
        self.n_dataset_classes = n_dataset_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)

        self.contextembed_digit1 = EmbedFC(n_digit_classes, 2 * n_feat)
        self.contextembed_digit2 = EmbedFC(n_digit_classes, 1 * n_feat)

        self.contextembed_dataset1 = EmbedFC(n_dataset_classes, 2 * n_feat)
        self.contextembed_dataset2 = EmbedFC(n_dataset_classes, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, digit_labels, dataset_labels, t, context_mask):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        digit_one_hot = nn.functional.one_hot(digit_labels, num_classes=self.n_digit_classes).type(torch.float)
        dataset_one_hot = nn.functional.one_hot(dataset_labels, num_classes=self.n_dataset_classes).type(torch.float)
        
        context_mask_digit = context_mask[:, None]
        context_mask_digit = context_mask_digit.repeat(1, self.n_digit_classes)
        context_mask_digit = (-1 * (1 - context_mask_digit)) 
        digit_one_hot = digit_one_hot * context_mask_digit

        context_mask_dataset = context_mask[:, None]
        context_mask_dataset = context_mask_dataset.repeat(1, self.n_dataset_classes)
        context_mask_dataset = (-1 * (1 - context_mask_dataset))
        dataset_one_hot = dataset_one_hot * context_mask_dataset
        
        cemb_digit1 = self.contextembed_digit1(digit_one_hot).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb_digit2 = self.contextembed_digit2(digit_one_hot).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        cemb_dataset1 = self.contextembed_dataset1(dataset_one_hot).view(-1, self.n_feat * 2, 1, 1)
        cemb_dataset2 = self.contextembed_dataset2(dataset_one_hot).view(-1, self.n_feat, 1, 1)

        combined_emb1 = cemb_digit1 + cemb_dataset1
        combined_emb2 = cemb_digit2 + cemb_dataset2

        up1 = self.up0(hiddenvec)
        up2 = self.up1(combined_emb1 * up1 + temb1, down2)
        up3 = self.up2(combined_emb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T):
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, digit_labels, dataset_labels):
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device) 
        noise = torch.randn_like(x)  

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  
        
        context_mask = torch.bernoulli(torch.ones_like(digit_labels, dtype=torch.float) * self.drop_prob).to(self.device)
        
        return self.loss_mse(noise, self.nn_model(x_t, digit_labels, dataset_labels, _ts / self.n_T, context_mask))
    
    def sample(self, n_sample, size, device, digit_labels, dataset_labels, guide_w=0.0):
        assert digit_labels.shape[0] == n_sample, "digit_labels 必須與 n_sample 的數量一致"
        assert dataset_labels.shape[0] == n_sample, "dataset_labels 必須與 n_sample 的數量一致"

        x_i = torch.randn(n_sample, *size).to(device)

        context_mask = torch.zeros_like(digit_labels, dtype=torch.float).to(device)

        digit_labels = digit_labels.repeat(2)
        dataset_labels = dataset_labels.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.0

        x_i_store = []
        print()

        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}  ', end='\r')
            
            t_is = torch.full((2 * n_sample,), i / self.n_T, dtype=torch.float).to(device)
            t_is = t_is.view(-1, 1, 1, 1) 

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            x_i = x_i.repeat(2, 1, 1, 1)  

            eps = self.nn_model(x_i, digit_labels, dataset_labels, t_is, context_mask)

            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]

            eps = (1 + guide_w) * eps1 - guide_w * eps2

            x_i = x_i[:n_sample]

            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store


def SettingSeed(seed = 6666):
    myseed = seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)


class HW2P1Dataset(Dataset):
    def __init__(self, path, tfm, what_dataset):
        super(HW2P1Dataset).__init__()

        self.image_file_folder = os.path.join(path, 'data') 
        self.csv_file_path = os.path.join(path, 'train.csv')
        
        df = pd.read_csv(self.csv_file_path)
        self.image_names = df['image_name'].tolist()
        self.digit_labels = df['label'].tolist()

        num = len(self.image_names)
        if what_dataset == 'mnistm':
            self.dataset_labels = [0] * num
        elif what_dataset == 'svhn':
            self.dataset_labels = [1] * num

        self.files = [os.path.join(self.image_file_folder, x) for x in self.image_names]

        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)

        return im, self.digit_labels[idx], self.dataset_labels[idx]


def train():

    SettingSeed(6666)

    writer = SummaryWriter()

    n_epochs = 300

    batch_size = 64

    n_T = 500

    device = "cuda" if torch.cuda.is_available() else "cpu"

    digit_classes = 10
    dataset_classes = 2

    n_feat = 256

    learning_rate = 1e-4

    save_dir_m = "p1-model"

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_digit_classes=digit_classes, n_dataset_classes=dataset_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1).to(device)

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),])

    train_set_1 = HW2P1Dataset("./hw2_data/digits_1/mnistm", tfm=tf, what_dataset='mnistm')
    train_set_2 = HW2P1Dataset("./hw2_data/digits_1/svhn", tfm=tf, what_dataset='svhn')

    combined_train_set = ConcatDataset([train_set_1, train_set_2])

    train_loader = DataLoader(combined_train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    optim = torch.optim.Adam(ddpm.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):

        ddpm.train()

        optim.param_groups[0]['lr'] = learning_rate * (1-epoch/n_epochs)

        for batch in tqdm(train_loader):
            
            optim.zero_grad()

            imgs, d_labels, s_labels = batch
            imgs = imgs.to(device)
            d_labels = d_labels.to(device)
            s_labels = s_labels.to(device)

            train_loss = ddpm(imgs, d_labels, s_labels)
            train_loss.backward()

            optim.step()
    
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")

        if (epoch+1) % 10 == 0:
            torch.save(ddpm.state_dict(), save_dir_m + f"/model_{epoch+1}.pth")

        
if __name__ == "__main__":
    train()

