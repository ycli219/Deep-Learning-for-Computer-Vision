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
import sys


# ResNet風格的卷積區塊，用於特徵提取。
class ResidualConvBlock(nn.Module):
    # in_channels: 輸入特徵圖的通道數。
    # out_channels: 輸出特徵圖的通道數。
    # is_res: 是否應用殘差連接。
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()

        # same_channels，用於檢查輸入和輸出通道數是否相同。這將影響殘差連接的方式。
        self.same_channels = in_channels==out_channels
        self.is_res = is_res

        # 使用 nn.Sequential定義了一個由三個層組成的序列
        # nn.Conv2d(in_channels, out_channels, 3, 1, 1): 2D卷積層，卷積核大小為 3x3，步幅為 1，填充為 1。
        # nn.BatchNorm2d(out_channels): 批歸一化層，用於穩定和加速訓練。
        # nn.GELU(): 激活函數，GELU是一種非線性激活函數，以增加模型的非線性表達能力。
        # 假設輸入數據的形狀為 (batch_size, in_channels, height, width)，經過一個 2D卷積層後，
        # 輸出數據的形狀將變為 (batch_size, out_channels, new_height, new_width)。
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

    # 定義前向傳播方法，接受一個張量 x作為輸入，並返回一個張量作為輸出。
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            
            # 如果輸入和輸出通道數相同，直接將輸入與經過兩個卷積層後的輸出相加。
            # 如果通道數不同，將經過第一個卷積層後的中間輸出與第二個卷積層後的輸出相加。
            # 最後，將結果除以 1.414以標準化，避免激活值過大或過小，從而穩定訓練過程。
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            # 只進行兩個卷積層的操作，不進行任何相加。
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


# UnetDown將作為 UNet架構中「下採樣」(downsampling)路徑的一部分，用於減少特徵圖的空間尺寸(高度和寬度)，同時增加特徵的抽象程度。
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()

        # 創建一個包含兩個層的列表
        # nn.MaxPool2d(2):
        # 使用 PyTorch的最大池化層(Max Pooling Layer)，窗口大小為 2x2，
        # 將特徵圖的空間尺寸(高度和寬度)縮小一半，從而實現下採樣(Downsampling)。
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()

        # nn.ConvTranspose2d: 轉置卷積層(也稱為反卷積層)，用於上採樣特徵圖。
        # kernel_size=2: 卷積核大小為 2x2。
        # stride=2: 步幅為 2，這將使得特徵圖的高度和寬度各自增加一倍。
        # 將輸入特徵圖的空間尺寸上採樣(例如，從 16x16上採樣到 32x32)。
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    # skip: 來自編碼器對應層的跳接特徵圖(跳接特徵圖用於融合高分辨率的空間信息)。
    # torch.cat: 將兩個張量在指定的維度上進行拼接。
    # 1: 指定拼接的維度為通道維度(在 PyTorch中，張量的維度順序通常為 (batch_size, channels, height, width))。
    # 將上採樣後的特徵圖 x與來自編碼器的跳接特徵圖 skip在通道維度上進行拼接，形成一個新的特徵圖。這樣做有助於保留高分辨率的空間信息，提升解碼器的表達能力。
    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


# EmbedFC用於將輸入的離散或連續數據(如標籤、時間步等)轉換為連續的嵌入向量，以便在模型中進行進一步的處理。
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
        # x.view(-1, self.input_dim):
        # 將輸入張量 x重塑為形狀(batch_size, input_dim)。
        # -1: 自動推斷這一維的大小，通常對應於批量大小 batch_size。
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    # in_channels: 輸入圖像的通道數(例如，RGB 圖像為 3)。
    # n_feat: 模型中使用的特徵通道數(默認為 256)。
    # n_digit_classes: 數字標籤的類別數(默認為 10，表示數字 0-9)。
    # n_dataset_classes: 數據集標籤的類別數(默認為 2，表示 MNIST-M 和 SVHN)。
    def __init__(self, in_channels, n_feat = 256, n_digit_classes=10, n_dataset_classes=2):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_digit_classes = n_digit_classes
        self.n_dataset_classes = n_dataset_classes

        # 使用 ResidualConvBlock類別，將輸入圖像的通道數從 in_channels轉換為 n_feat。
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        
        # 提取全局特徵: 通過全局平均池化和激活函數，將編碼器最後一層的特徵圖轉換為全局特徵向量，這個過程有助於捕捉整個圖像的全局上下文信息，而不僅僅是局部特徵。
        # 信息壓縮與聚合: 壓縮特徵圖的空間尺寸，減少計算量。
        # 連接編碼器和解碼器: 作為編碼器和解碼器之間的橋樑，將編碼器提取的特徵傳遞給解碼器，並與嵌入的條件信息進行融合。
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        # 使用 EmbedFC類別將時間步 t嵌入為連續向量:
        # self.timeembed1 將時間步從 1維轉換為 2 * n_feat維。
        # self.timeembed2 將時間步從 1維轉換為 1 * n_feat維。
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)

        # 使用 EmbedFC類別將數字標籤 digit_labels嵌入為連續向量:
        self.contextembed_digit1 = EmbedFC(n_digit_classes, 2 * n_feat)
        self.contextembed_digit2 = EmbedFC(n_digit_classes, 1 * n_feat)

        # 使用 EmbedFC 將數據集標籤 dataset_labels嵌入為連續向量:
        self.contextembed_dataset1 = EmbedFC(n_dataset_classes, 2 * n_feat)
        self.contextembed_dataset2 = EmbedFC(n_dataset_classes, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        # in_channels = 4 * n_feat: 這通常來自於拼接了上採樣特徵圖和跳接特徵圖。
        # (例如，2 * n_feat + 2 * n_feat)。
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
        # 將 down1通過第二個下採樣模組 UnetDown，下採樣空間尺寸並增加特徵通道數。輸出形狀變為 (batch_size, 2 * n_feat, H/4, W/4)。
        down2 = self.down2(down1)
        # 形狀變為 (batch_size, 2 * n_feat, 1, 1)。
        hiddenvec = self.to_vec(down2)

        digit_one_hot = nn.functional.one_hot(digit_labels, num_classes=self.n_digit_classes).type(torch.float)
        dataset_one_hot = nn.functional.one_hot(dataset_labels, num_classes=self.n_dataset_classes).type(torch.float)
        
        '''
        舉例
        初始 context_mask: [1, 0, 1]
        1. [[1], [0], [1]] #(3, 1)
        2. #(3, 10)
        3. [[ 0,  0,  0, ..., 0],
            [-1, -1, -1, ..., -1],
            [ 0,  0,  0, ..., 0]] mask

           [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 標籤2
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],   # 標籤5
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]   # 標籤7

        4. 乘積
           [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0,-1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        context_mask = 1: 將數字標籤和數據集標籤設為 0，即遮蔽這些標籤，模型在生成圖像時不會依賴這些條件信息。
        context_mask = 0: 將數字標籤和數據集標籤乘以 -1，即保留這些標籤。這裡使用 -1 可能是為了區分於遮蔽的情況，具體應用時可以通過其他部分的代碼來處理這些負值標籤。
        '''
        context_mask_digit = context_mask[:, None]
        context_mask_digit = context_mask_digit.repeat(1, self.n_digit_classes)
        context_mask_digit = (-1 * (1 - context_mask_digit))  # 將 0和 1翻轉
        digit_one_hot = digit_one_hot * context_mask_digit

        context_mask_dataset = context_mask[:, None]
        context_mask_dataset = context_mask_dataset.repeat(1, self.n_dataset_classes)
        context_mask_dataset = (-1 * (1 - context_mask_dataset))
        dataset_one_hot = dataset_one_hot * context_mask_dataset
        
        # 將 digit_one_hot通過 self.contextembed_digit1轉換為形狀 (batch_size, 2 * n_feat)的嵌入向量，然後重塑為 (batch_size, 2 * n_feat, 1, 1)。
        cemb_digit1 = self.contextembed_digit1(digit_one_hot).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb_digit2 = self.contextembed_digit2(digit_one_hot).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        cemb_dataset1 = self.contextembed_dataset1(dataset_one_hot).view(-1, self.n_feat * 2, 1, 1)
        cemb_dataset2 = self.contextembed_dataset2(dataset_one_hot).view(-1, self.n_feat, 1, 1)

        # 形狀為 (batch_size, 2 * n_feat, 1, 1)。
        combined_emb1 = cemb_digit1 + cemb_dataset1
        # 形狀為 (batch_size, n_feat, 1, 1)。
        combined_emb2 = cemb_digit2 + cemb_dataset2

        up1 = self.up0(hiddenvec)
        # 將 up1和 combined_emb1逐元素相乘，並加上時間步嵌入 temb1。
        # 將結果傳遞給 self.up1，並將編碼器部分的 down2作為跳接傳遞進去。
        # UnetUp 將這些特徵圖拼接並進行上採樣和特徵提取。
        up2 = self.up1(combined_emb1 * up1 + temb1, down2)
        up3 = self.up2(combined_emb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T):
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    # torch.arange(0, T + 1, dtype=torch.float32) :
    # 這段產生了一個從0到T(包含T)的整數序列，並將其轉換為浮點數類型的張量。
    # 這代表擴散過程中的每個時間步，總共有T+1個步驟。

    # (beta2 - beta1) : 
    # 這段定義了噪聲參數的範圍，從beta1到beta2。這兩個值是用來控制噪聲變異數的下限和上限。

    # torch.arange(0, T + 1, dtype=torch.float32) / T :
    # 將步驟序列除以總步數T，這樣會將序列標準化，使其從0線性增加到1。

    # (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T :
    # 將標準化的時間步序列乘以(beta2 - beta1)，這樣每個時間步的變化量都在beta1和beta2之間均勻分佈。
 
    # + beta1 :
    # 最後，加上beta1，使得每個時間步的噪聲係數從beta1開始，並逐步增加到beta2。
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

        # register_buffer是 PyTorch的一個方法，用於註冊不需要梯度更新的張量。這些張量會被保存到模型的狀態字典中，但不會在反向傳播時更新。
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, digit_labels, dataset_labels):
        # 使用 torch.randint隨機抽取一個介於 1和 n_T(包含 1和 n_T)之間的整數，生成一個形狀為 (batch_size,)的張量 _ts，表示每個樣本的時間步。
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        # 使用 torch.randn_like生成一個與 x形狀相同的隨機噪聲張量 noise，其元素服從標準正態分佈(均值為 0，方差為 1)。
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        # [_ts, None, None, None]用於選擇對應時間步的係數，並擴展維度以便與 x和 noise進行逐元素相乘。
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  

        # torch.ones_like(digit_labels, dtype=torch.float): 生成一個與 digit_labels (dataset_labels同)形狀相同、所有元素為 1的張量。假設 digit_labels的形狀為 (batch_size,)，則生成的張量形狀也是 (batch_size,)。
        # * self.drop_prob: 將每個元素乘以 drop_prob(例如，0.1)，生成一個全為 drop_prob的張量。
        # torch.bernoulli(...): 每個位置有 drop_prob的概率被遮蔽(mask為 1)，有 1 - drop_prob的概率保留(mask為 0)。
        context_mask = torch.bernoulli(torch.ones_like(digit_labels, dtype=torch.float) * self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, digit_labels, dataset_labels, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, digit_labels, dataset_labels, guide_w=0.0):
        """
        Generate samples using the DDPM model with classifier-free guidance.
        
        Args:
            n_sample (int): Number of samples to generate.
            size (tuple): Shape of each sample (channels, height, width).
            device (torch.device): Device to perform computation on.
            digit_labels (torch.Tensor): Tensor of digit labels for conditioning (shape: [n_sample]).
            dataset_labels (torch.Tensor): Tensor of dataset labels for conditioning (shape: [n_sample]).
            guide_w (float): Guidance scale for classifier-free guidance.

        Returns:
            torch.Tensor: Generated samples.
            np.ndarray: Generated steps (for visualization).
        """
        assert digit_labels.shape[0] == n_sample, "digit_labels 必須與 n_sample 的數量一致"
        assert dataset_labels.shape[0] == n_sample, "dataset_labels 必須與 n_sample 的數量一致"

        # 生成初始噪聲 x_T ~ N(0, 1)
        x_i = torch.randn(n_sample, *size).to(device)

        # 測試時不進行上下文遮蔽。
        context_mask = torch.zeros_like(digit_labels, dtype=torch.float).to(device)

        # 擴展批次: 將批次大小翻倍，為無條件和有條件的生成做準備。
        digit_labels = digit_labels.repeat(2)
        dataset_labels = dataset_labels.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.0  # 第二半部分的批次為無條件生成(context_mask=1)

        # 保存生成過程中的步驟(可選，用於可視化)。
        x_i_store = []
        print()

        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}  ', end='\r')
            
            # 計算當前時間步的歸一化值。
            t_is = torch.full((2 * n_sample,), i / self.n_T, dtype=torch.float).to(device)
            t_is = t_is.view(-1, 1, 1, 1)  # 形狀轉換為 (2*n_sample, 1, 1, 1)。

            # 生成噪聲 z ~ N(0, 1) 如果不是最後一步。
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # 前向傳播時批次擴展。
            x_i = x_i.repeat(2, 1, 1, 1)  # 擴展批次大小為 2*n_sample

            # 預測噪聲 eps
            eps = self.nn_model(x_i, digit_labels, dataset_labels, t_is, context_mask)

            # 將批次分割回原來的大小。
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]

            # 應用指導比例 w 進行混合。
            eps = (1 + guide_w) * eps1 - guide_w * eps2

            # 恢復原始批次大小。
            x_i = x_i[:n_sample]

            # 更新 x_i 根據 DDPM 的反向過程公式。
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

            # 保存生成步驟（可選）。
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        # 將生成步驟轉換為 NumPy 陣列（可選）。
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


path_of_output_folder = "different-timesteps"


def generate_imgs():

    SettingSeed(5151)

    n_T = 500

    device = "cuda" if torch.cuda.is_available() else "cpu"

    digit_classes = 10
    dataset_classes = 2

    datasets = {
        'mnistm': 0,
        'svhn': 1
    }
    digits = list(range(10))  # 0-9 digits

    n_feat = 256

    batch_size = 50

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_digit_classes=digit_classes, n_dataset_classes=dataset_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1).to(device)
    ddpm.load_state_dict(torch.load("./H2O/p1-model-4/model_40.pth"))

    ddpm.eval()
    with torch.no_grad():

        for dataset_name, dataset_label in datasets.items():
                
            for digit in digits:

                print(f'Generating {batch_size} images for {dataset_name} with digit {digit}...')

                digit_labels = torch.full((batch_size,), digit, dtype=torch.long).to(device)
                dataset_labels_tensor = torch.full((batch_size,), dataset_label, dtype=torch.long).to(device)

                _, x_i_store = ddpm.sample(
                    n_sample=batch_size,
                    size=(3, 28, 28),
                    device=device,
                    digit_labels=digit_labels,
                    dataset_labels=dataset_labels_tensor,
                    guide_w=2.0  # 根據需要調整指導比例
                )
                
                store_dir = os.path.join(path_of_output_folder, dataset_name, 'x_i_store')
                os.makedirs(store_dir, exist_ok=True)

                for step_idx, step_images in enumerate(x_i_store):
                    step_images = torch.tensor(step_images)
                    for img_idx in range(step_images.shape[0]):
                        img = step_images[img_idx]
                        img = (img + 1) / 2
                        img = torch.clamp(img, 0, 1)
                        img_path = os.path.join(store_dir, f'step_{step_idx + 1:03d}_digit_{digit}_{img_idx + 1:03d}.png')
                        if digit == 0 and img_idx == 0:
                            save_image(img, img_path)


if __name__ == "__main__":
    generate_imgs()