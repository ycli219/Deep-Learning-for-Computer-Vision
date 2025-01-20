import os, sys
import cv2
import torch
import numpy as np
import json
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from transformers import AutoFeatureExtractor


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def main():

    # Parse arguments using sys.argv
    json_file = sys.argv[1]
    outdir =  sys.argv[2]
    ckpt =  sys.argv[3]
    config = "./stable-diffusion/configs/stable-diffusion/v1-inference.yaml"

    # Load the pretrained model
    config = OmegaConf.load(f"{config}")
    model = load_model_from_config(config, f"{ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # Load additional token embeddings
    tokenizer = model.cond_stage_model.tokenizer
    new_tokens = ["<new1>", "<new2>"]
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    if num_added_tokens > 0:
        model.cond_stage_model.transformer.resize_token_embeddings(len(tokenizer))
        embedding_layer = model.cond_stage_model.transformer.get_input_embeddings()
        embedding_files = {
            "<new1>": 'embedding_new1_epoch_100.pth',
            "<new2>": 'embedding_new2_epoch_100.pth'
        }
        for new_token in new_tokens:
            embedding_dict = torch.load(embedding_files[new_token])
            for token, embedding in zip(embedding_dict['tokens'], embedding_dict['embeddings']):
                token_id = tokenizer.convert_tokens_to_ids(token)
                embedding_layer.weight.data[token_id] = embedding.to(device)

    # Load testing prompts from JSON file
    with open(json_file, 'r') as f:
        prompts_data = json.load(f)

    # Iterate over each prompt in the JSON
    precision = "autocast"
    precision_scope = autocast if precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        for key, data in prompts_data.items():
            src_image = data['src_image']
            prompts = data['prompt']
            token_name = data['token_name']

            seed_everything(62)

            for prompt_idx, prompt in enumerate(prompts):
                # Create output folder structure
                prompt_folder = os.path.join(outdir, key, str(prompt_idx))
                os.makedirs(prompt_folder, exist_ok=True)

                for img_idx in range(25):
                    # Generate image using the model with the specified prompt
                    with torch.no_grad():
                        uc = model.get_learned_conditioning([""])
                        c = model.get_learned_conditioning([prompt])
                        shape = [4, 64, 64]  # Example shape, adjust as needed
                        samples_ddim, _ = DPMSolverSampler(model).sample(S=50, conditioning=c, batch_size=1, shape=shape, verbose=False, unconditional_guidance_scale=7.5, unconditional_conditioning=uc, eta=0.0, x_T=None)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                    
                    for x_sample in x_checked_image_torch:
                        # Post-process the generated image if needed (e.g., converting to a PIL image)
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        pil_image = Image.fromarray(x_sample.astype(np.uint8))
                        
                        # Save the generated image
                        output_image_path = os.path.join(prompt_folder, f"source{int(key)}_prompt{prompt_idx}_{img_idx}.png")
                        pil_image.save(output_image_path)
                    

if __name__ == "__main__":
    main()
