import os
import json
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tokenizer import BPETokenizer
from p2_encoder import ImageCaptioningEncoder
from p2_decoder import ModifiedDecoder, Config
import timm
from tqdm import tqdm
import sys

path_to_folder_of_test_images = sys.argv[1]
path_to_output_file =  sys.argv[2]
path_to_decoder_weights = sys.argv[3]

class InferenceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        self.image_files = []
        for filename in sorted(os.listdir(image_dir)):
            if filename.endswith('.jpg'):
                self.image_files.append(filename)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, os.path.splitext(filename)[0]

import loralib as lora
def load_trained_parameters(decoder, checkpoint_path):
    """載入訓練好的參數"""
    checkpoint = torch.load(checkpoint_path)
    decoder.load_state_dict(checkpoint['decoder_params'], strict=False)

    lora.mark_only_lora_as_trainable(decoder)
    for n, p in decoder.named_parameters():
        if 'projection' in n:
            p.requires_grad = True
    
    print(sum(p.numel() for p in decoder.parameters() if p.requires_grad))

def generate_captions_batch(encoder, decoder, tokenizer, dataloader, max_length=64, device='cuda'):
    encoder.eval()
    decoder.eval()
    
    end_token = tokenizer.encoder["<|endoftext|>"]
    results = {}
    
    with torch.no_grad():
        for batch_images, batch_filenames in tqdm(dataloader, desc="Generating captions"):
            batch_images = batch_images.to(device)
            visual_features = encoder(batch_images)
            current_ids = torch.full((len(batch_images), 1), end_token, 
                                  device=device)
            
            # 逐步生成
            for _ in range(max_length):
                logits = decoder(visual_features, current_ids)
                next_token_logits = logits[:, -1, :]
                next_tokens = next_token_logits.argmax(dim=-1).unsqueeze(1)
                current_ids = torch.cat([current_ids, next_tokens], dim=1)
            
            # 解碼生成的序列
            for ids, filename in zip(current_ids, batch_filenames):
                # 找到結束符的位置
                end_positions = (ids == end_token).nonzero(as_tuple=True)[0]
                if len(end_positions) > 1:  # 有找到第二個結束符
                    end_pos = end_positions[1].item()
                else: 
                    end_pos =  30
                    
                caption_ids = ids[1:end_pos].tolist()  # 從第二個token開始（跳過起始token）
                caption = tokenizer.decode(caption_ids)
                results[filename] = caption
                print(filename)
                print(caption)
    
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_workers = 4
    
    tokenizer = BPETokenizer(
        encoder_file='encoder.json',
        vocab_file='vocab.bpe'
    )
    
    encoder = ImageCaptioningEncoder().to(device)
    decoder_cfg = Config(checkpoint=path_to_decoder_weights)
    decoder = ModifiedDecoder(decoder_cfg, lora_rank=58).to(device)
    
    load_trained_parameters(decoder, './best_model.pth')
    
    model = timm.create_model('vit_large_patch14_clip_224.openai_ft_in1k', pretrained=True, num_classes=0)
    model = model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    
    transform_list = [
        timm.data.create_transform(**data_config, is_training=False)
    ]
    transform = transforms.Compose(transform_list)

    test_dataset = InferenceDataset(
        image_dir=path_to_folder_of_test_images,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    results = generate_captions_batch(encoder, decoder, tokenizer, test_loader, device=device)
    
    output_path = path_to_output_file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results have been saved to {output_path}")

if __name__ == '__main__':
    main()