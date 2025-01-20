import torch
import torchvision
from PIL import Image
import numpy as np
import os
from tqdm.auto import tqdm
from torchvision.utils import save_image, make_grid

import UNet
import utils

class DDIM:
    def __init__(self, eta=0.0, noise_steps=50, device=None):
        self.eta = eta
        self.total_steps = noise_steps
        self.device = device
        # Define beta scheduler and compute alpha values
        self.beta = utils.beta_scheduler().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)

    def sample(self, model, noise_filename):
        # Define the sequence of timesteps
        timestep_sequence = np.arange(0, 1000, 20) + 1
        previous_timestep_sequence = np.insert(timestep_sequence[:-1], 0, 0)

        model.eval()

        with torch.no_grad():
            # Load the predefined noise tensor
            latent = torch.load(os.path.join("./hw2_data/face/noise", f"{noise_filename}.pt")).to(self.device)

            # Perform reverse iterations to generate the image
            for step_idx in tqdm(reversed(range(1, self.total_steps))):
                current_step = torch.tensor([timestep_sequence[step_idx]], device=self.device, dtype=torch.long)
                previous_step = torch.tensor([previous_timestep_sequence[step_idx]], device=self.device, dtype=torch.long)

                # Predict the noise using the model
                predicted_noise = model(latent, current_step)

                # Retrieve current and previous alpha values
                alpha_current = self.alpha_cumulative[current_step].view(-1, 1, 1, 1)
                alpha_previous = self.alpha_cumulative[previous_step].view(-1, 1, 1, 1)

                # Compute sigma
                sigma = self.eta * torch.sqrt((1 - alpha_previous) / (1 - alpha_current)) * torch.sqrt(1 - (alpha_current / alpha_previous))

                # Generate random noise
                random_noise = torch.randn_like(latent)

                # Update the latent representation
                latent = torch.sqrt(alpha_previous) * ((latent - torch.sqrt(1 - alpha_current) * predicted_noise) / torch.sqrt(alpha_current))
                latent += torch.sqrt(1 - alpha_previous - sigma ** 2) * predicted_noise
                latent += sigma * random_noise

                # Ensure the data type is float
                latent = latent.float()

            # Clamp the latent representation to a specific range
            latent = latent.clamp(-1, 1)

        model.train()

        return latent


def create_grid_image(path_directory_generated_images, predefined_noises_files, eta_values, output_path):
    all_images = []
    
    for eta in eta_values:
        col_images = []
        for file in predefined_noises_files:
            # 加載生成的圖片
            img_path = os.path.join(path_directory_generated_images, f"{file}_eta_{eta}.png")
            img = Image.open(img_path)
            img_tensor = torchvision.transforms.ToTensor()(img)  # 轉換為 tensor
            col_images.append(img_tensor)  # 加入該行
        all_images.append(torch.stack(col_images))  # 將這一行所有圖片加入
    
    # 將所有行合併為一個大網格
    grid = make_grid(torch.cat(all_images, dim=0), nrow=len(predefined_noises_files), padding=2, normalize=True)

    # 保存網格圖片
    save_image(grid, output_path)
    print(f'Saved grid image at {output_path}')


def SettingSeed(seed = 6666):
    myseed = seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet.UNet().to(device)
    model.load_state_dict(torch.load("./hw2_data/face/UNet.pt"))

    predefined_noises_files = ['00', '01', '02', '03']

    eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    for file in predefined_noises_files:

        for eta in eta_values:
            
            ddim = DDIM(eta=eta, noise_steps=50, device=device)

            x = ddim.sample(model, file)

            save_image(x, os.path.join("./p2-different-eta", f"{file}_eta_{eta}.png"), normalize=True)

            print(f'Saved image at {os.path.join("./p2-different-eta", f"{file}_eta_{eta}.png")}')

    output_path = os.path.join("./p2-different-eta", 'grid_image.png')
    create_grid_image("./p2-different-eta", predefined_noises_files, eta_values, output_path)