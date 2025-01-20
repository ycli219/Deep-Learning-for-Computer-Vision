import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
import random
from tqdm.auto import tqdm
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from ldm.util import instantiate_from_config
from scripts.txt2img import load_model_from_config
from transformers import CLIPModel, CLIPProcessor


myseed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


# Load profiles and models
config = OmegaConf.load("./stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
ckpt = "./stable-diffusion/ldm/models/stable-diffusion-v1/model.ckpt"

model = load_model_from_config(config, ckpt)

# Define image paths and prompts
image_paths = [
    "../../../H2O/hw2_data/textual_inversion/0/04.jpg",
    "../../../H2O/hw2_data/textual_inversion/0/03.jpg",
    "../../../H2O/hw2_data/textual_inversion/0/02.jpg",
    "../../../H2O/hw2_data/textual_inversion/0/01.jpg",
    "../../../H2O/hw2_data/textual_inversion/0/00.jpg"
]

general_prompts = [
    "a photo of a <new1>",
    "a rendering of a <new1>",
    "a cropped photo of the <new1>",
    "the photo of a <new1>",
    "a photo of my <new1>",
    "a photo of the cute <new1>",
    "a close-up photo of a <new1>",
    "a bright photo of the <new1>",
    "a cropped photo of a <new1>",
    "a photo of the <new1>",
    "a good photo of the <new1>",
    "a photo of one <new1>",
    "a close-up photo of the <new1>",
    "a rendition of the <new1>",
    "a rendition of a <new1>",
    "a photo of a nice <new1>",
    "a good photo of a <new1>",
    "a photo of the nice <new1>",
    "a photo of a cute <new1>"
]

transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop((512, 512)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Get tokenizer from model
tokenizer = model.cond_stage_model.tokenizer

# Add new marker words
new_tokens = ["<new1>"]
num_added_tokens = tokenizer.add_tokens(new_tokens)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if num_added_tokens > 0:
    # Adjust the size of the model's embedding matrix
    model.cond_stage_model.transformer.resize_token_embeddings(len(tokenizer))

    # Initialize new word embeddings
    token_embeds = model.cond_stage_model.transformer.get_input_embeddings().weight.data
    for token in new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)

        # Initialize with word embeddings of 'dog'
        token_embeds[token_id] = token_embeds[tokenizer.convert_tokens_to_ids('dog')].clone()

        
# Ensure that the embedding matrix requires_grad is True
embedding_layer = model.cond_stage_model.transformer.get_input_embeddings()
embedding_layer.weight.requires_grad = True  # The entire embedding matrix

# Configure the optimizer to pass only the parameters of the embedding layer
optimizer = torch.optim.AdamW([embedding_layer.weight], lr=5e-4)

# Define dataset classes
class TextualInversionDataset(Dataset):
    def __init__(self, image_paths, general_prompts, tokenizer, transform=None, repeats=100):
        self.image_paths = image_paths
        self.general_prompts = general_prompts
        self.tokenizer = tokenizer
        self.transform = transform
        self.num_images = len(self.image_paths)
        self._length = self.num_images * repeats

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        image_idx = idx % self.num_images
        image = Image.open(self.image_paths[image_idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        prompt = random.choice(self.general_prompts)
        
        return {"image": image, "prompt": prompt}


# Create datasets and data loaders
dataset = TextualInversionDataset(
    image_paths=image_paths,
    general_prompts=general_prompts,
    tokenizer=tokenizer,
    transform=transform,
    repeats=1
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

writer = SummaryWriter()

model = model.to(device)

num_epochs = 100

for epoch in range(num_epochs):
    for batch in tqdm(dataloader):
        images = batch["image"].to(device)
        prompts = batch["prompt"]

        optimizer.zero_grad()

        # forward propagation
        # 1. Encoding images as potential representations
        encoder_posterior = model.encode_first_stage(images)
        z = model.get_first_stage_encoding(encoder_posterior)

        # 2. Adding noise to the potential representation
        batch_size = images.size(0)
        t = torch.randint(0, model.num_timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(z)
        z_noisy = model.q_sample(z, t, noise=noise)

        # 3. Obtaining text condition codes
        c = model.get_learned_conditioning(prompts)

        # 4. Computational model output
        model_output = model.apply_model(z_noisy, t, c)

        # Calculated losses
        loss = torch.nn.functional.mse_loss(model_output, noise)

        # Backpropagation and optimization
        loss.backward()

        optimizer.step()

    if (epoch + 1) <= 100:

        # Save the current word embedding
        token_embeds = embedding_layer.weight.data
        new_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in new_tokens]
        new_embeddings = token_embeds[new_token_ids, :].cpu()

        # Create a dictionary to save word embeddings
        embedding_dict = {
            'token_ids': new_token_ids,
            'embeddings': new_embeddings,
            'tokens': new_tokens  
        }

        embedding_save_path = os.path.join("./embed5", f"embedding_new1_epoch_{epoch+1}.pth") 
        torch.save(embedding_dict, embedding_save_path)
        print(f"Embeddings saved at epoch {epoch+1}")

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    writer.add_scalar("Loss/Train", loss, epoch+1)

    torch.cuda.empty_cache()
    import gc
    gc.collect()
