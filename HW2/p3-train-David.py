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


myseed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

config = OmegaConf.load("./stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
ckpt = "./stable-diffusion/ldm/models/stable-diffusion-v1/model.ckpt"

model = load_model_from_config(config, ckpt)

image_paths = [
    "../../../H2O/hw2_data/textual_inversion/1/2018-07-01_Adventure_by-David-Revoy.jpg",
    "../../../H2O/hw2_data/textual_inversion/1/2019-11-08_cover_book-project.jpg",
    "../../../H2O/hw2_data/textual_inversion/1/2022-05-31_ep37-panel1-rendered_net.jpg",
    "../../../H2O/hw2_data/textual_inversion/1/plxthumbnailer.jpg",
    "../../../H2O/hw2_data/textual_inversion/1/plxthumbnailer-1.jpg"
]

general_prompts = [
    "A painting in the style of <new2>.",
    "A rendering in the style of <new2>.",
    "A cropped painting in the style of <new2>.",
    "The painting in the style of <new2>.",
    "A clean painting in the style of <new2>.",
    "A picture in the style of <new2>.",
    "A cool painting in the style of <new2>.",
    "A close-up painting in the style of <new2>.",
    "A bright painting in the style of <new2>.",
    "A cropped painting in the style of <new2>.",
    "A good painting in the style of <new2>.",
    "A close-up painting in the style of <new2>.",
    "A rendition in the style of <new2>.",
    "A nice painting in the style of <new2>.",
    "A small painting in the style of <new2>.",
    "A large painting in the style of <new2>."
]


transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop((512, 512)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

tokenizer = model.cond_stage_model.tokenizer

new_tokens = ["<new2>"]
num_added_tokens = tokenizer.add_tokens(new_tokens)
'''
#================
# 打印所有 tokenizer 中的 tokens 和对应的 ID
def print_all_tokens(tokenizer, save_to_file=False, file_path="tokenizer_vocab.txt"):
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])  # 按照 ID 排序
    
    if save_to_file:
        with open(file_path, "w", encoding="utf-8") as f:
            for token, id in sorted_vocab:
                f.write(f"ID: {id}, Token: {token}\n")
        print(f"所有 tokens 已保存到 {file_path}")
    else:
        for token, id in sorted_vocab:
            print(f"ID: {id}, Token: {token}")

# 调用函数打印 tokens
print_all_tokens(tokenizer, save_to_file=True)  # 如果词汇表很大，建议将 `save_to_file` 设置为 True
#================
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if num_added_tokens > 0:
    model.cond_stage_model.transformer.resize_token_embeddings(len(tokenizer))

    token_embeds = model.cond_stage_model.transformer.get_input_embeddings().weight.data
    for token in new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)

        token_embeds[token_id] = token_embeds[tokenizer.convert_tokens_to_ids('fantasy')].clone()

        
embedding_layer = model.cond_stage_model.transformer.get_input_embeddings()
embedding_layer.weight.requires_grad = True 

optimizer = torch.optim.AdamW([embedding_layer.weight], lr=5e-4)

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

        encoder_posterior = model.encode_first_stage(images)
        z = model.get_first_stage_encoding(encoder_posterior)

        batch_size = images.size(0)
        t = torch.randint(0, model.num_timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(z)
        z_noisy = model.q_sample(z, t, noise=noise)

        c = model.get_learned_conditioning(prompts)

        model_output = model.apply_model(z_noisy, t, c)

        loss = torch.nn.functional.mse_loss(model_output, noise)

        loss.backward()

        optimizer.step()

    if (epoch + 1) <= 100:
        token_embeds = embedding_layer.weight.data
        new_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in new_tokens]
        new_embeddings = token_embeds[new_token_ids, :].cpu()

        embedding_dict = {
            'token_ids': new_token_ids,
            'embeddings': new_embeddings,
            'tokens': new_tokens
        }

        embedding_save_path = os.path.join("./embed6", f"embedding_new2_epoch_{epoch+1}.pth") 
        torch.save(embedding_dict, embedding_save_path)
        print(f"Embeddings saved at epoch {epoch+1}")

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    writer.add_scalar("Loss/Train", loss, epoch+1)

    torch.cuda.empty_cache()
    import gc
    gc.collect()
