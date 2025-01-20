import torch
import torchvision
from torchvision.utils import save_image, make_grid
import numpy as np
import os
from tqdm.auto import tqdm
import UNet
import utils


def spherical_linear_interpolation(x0, x1, alpha):
    # 球形線性插值公式
    theta = torch.acos(torch.sum(x0 * x1) / (torch.norm(x0) * torch.norm(x1)))
    return (torch.sin((1 - alpha) * theta) / torch.sin(theta)) * x0 + (torch.sin(alpha * theta) / torch.sin(theta)) * x1


def linear_interpolation(x0, x1, alpha):
    # 線性插值公式
    return (1 - alpha) * x0 + alpha * x1


class DDIM:
    def __init__(self, eta=0.0, noise_steps=50, device=None):
        self.eta = eta
        self.total_steps = noise_steps
        self.device = device
        # Define beta scheduler and compute alpha values
        self.beta = utils.beta_scheduler().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)

    def sample(self, model, latent):
        # Define the sequence of timesteps
        timestep_sequence = np.arange(0, 1000, 20) + 1
        previous_timestep_sequence = np.insert(timestep_sequence[:-1], 0, 0)

        model.eval()

        with torch.no_grad():

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


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet.UNet().to(device)
    model.load_state_dict(torch.load("./hw2_data/face/UNet.pt"))

    ddim = DDIM(eta=0, noise_steps=50, device=device)

    # 加載預定義噪聲 00.pt 和 01.pt
    noise_00 = torch.load(os.path.join("./hw2_data/face/noise", "00.pt")).to(device)
    noise_01 = torch.load(os.path.join("./hw2_data/face/noise", "01.pt")).to(device)

    # 設定 alpha 值
    alphas = np.linspace(0, 1, 11)  # 0.0, 0.1, 0.2, ..., 1.0

    # 生成 SLERP 和 LERP 的插值圖像
    slerp_images = []
    lerp_images = []

    for alpha in alphas:
        # SLERP
        interpolated_noise_slerp = spherical_linear_interpolation(noise_00, noise_01, alpha)
        generated_image_slerp = ddim.sample(model, interpolated_noise_slerp)
        slerp_images.append(generated_image_slerp)

        # LERP
        interpolated_noise_lerp = linear_interpolation(noise_00, noise_01, alpha)
        generated_image_lerp = ddim.sample(model, interpolated_noise_lerp)
        lerp_images.append(generated_image_lerp)

    # 保存 SLERP 和 LERP 圖片序列
    slerp_grid = make_grid(torch.cat(slerp_images), nrow=len(alphas), padding=2, normalize=True)
    lerp_grid = make_grid(torch.cat(lerp_images), nrow=len(alphas), padding=2, normalize=True)

    save_image(slerp_grid, os.path.join("./p2-interpolation", 'slerp_grid.png'))
    save_image(lerp_grid, os.path.join("./p2-interpolation", 'lerp_grid.png'))

    print('Saved SLERP grid image at', os.path.join("./p2-interpolation", 'slerp_grid.png'))
    print('Saved LERP grid image at', os.path.join("./p2-interpolation", 'lerp_grid.png'))