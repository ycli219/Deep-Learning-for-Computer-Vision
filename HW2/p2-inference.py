import torch
import torchvision
from PIL import Image
import numpy as np
import os
from tqdm.auto import tqdm
import sys
from torchvision.utils import save_image

import UNet
import utils

noise_input_path = sys.argv[1]
generated_output_path = sys.argv[2]
pretrained_model_path = sys.argv[3]

class DDIMGenerator:
    def __init__(self, eta=0.0, steps=50, device=None):
        self.eta = eta
        self.total_steps = steps
        self.device = device
        # Define beta scheduler and compute alpha values
        self.beta = utils.beta_scheduler().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)

    def sample_image(self, model, noise_filename):
        # Define the sequence of timesteps
        timestep_sequence = np.arange(0, 1000, 20) + 1
        previous_timestep_sequence = np.insert(timestep_sequence[:-1], 0, 0)

        model.eval()

        with torch.no_grad():
            # Load the predefined noise tensor
            noise_path = os.path.join(noise_input_path, f"{noise_filename}.pt")
            latent = torch.load(noise_path).to(self.device)

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

def main():
    # Determine the computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained UNet model
    network = UNet.UNet().to(device)
    network.load_state_dict(torch.load(pretrained_model_path))

    # Initialize the DDIM generator
    generator = DDIMGenerator(eta=0.0, steps=50, device=device)

    # Retrieve all noise files (remove file extensions)
    noise_files = [os.path.splitext(fname)[0] for fname in os.listdir(noise_input_path) if fname.endswith('.pt')]

    # Iterate over each noise file to generate and save corresponding images
    for fname in noise_files:
        # Generate the image tensor
        generated_latent = generator.sample_image(network, fname)
        # Define the output file path
        output_file = os.path.join(generated_output_path, f"{fname}.png")
        # Save the generated image
        save_image(generated_latent, output_file, normalize=True)
        print(f"Image saved to {output_file}")

if __name__ == "__main__":
    main()