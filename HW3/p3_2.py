import math
import collections
import torch
from torch import nn
import torch.nn.functional as F
import loralib as lora
import os
import json
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tokenizer import BPETokenizer
from p2_encoder import ImageCaptioningEncoder
import timm


from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np


class ModifiedDecoder(nn.Module):
    def __init__(self, cfg, lora_rank=58):
        super().__init__()
        self.cfg = cfg

        # Projection layer (移自 encoder)
        vit_hidden_size = 1024  # ViT-Large 的輸出維度
        self.projection = nn.Sequential(
            nn.Linear(vit_hidden_size, cfg.n_embd),
            nn.ReLU(),
            nn.Linear(cfg.n_embd, cfg.n_embd)
        )
        
        # Transformer components
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block(cfg, lora_rank) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        
        # Output projection
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Tie weights
        self.transformer.wte.weight = self.lm_head.weight
        
        # Load checkpoint if provided
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = ['.c_attn.weight', '.c_fc.weight', '.c_proj.weight']
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)
            
    def forward(self, visual_features, text_ids):
        """
        Args:
            visual_features: 視覺特徵 [batch_size, num_patches, hidden_size]
            text_ids: 文字輸入 [batch_size, seq_len] (包含任何特殊token)
        Returns:
            logits: 每個位置的下一個token預測 [batch_size, seq_len, vocab_size]
        """
        device = visual_features.device

        # Project visual features
        visual_features = self.projection(visual_features)  # [batch_size, num_patches, n_embd]
        
        # 1. 獲取文字embeddings
        text_embeddings = self.transformer.wte(text_ids)
        
        # 2. 將視覺特徵和文字embeddings拼接
        combined_embeddings = torch.cat([visual_features, text_embeddings], dim=1)
        
        # 3. 生成位置編碼
        total_seq_len = combined_embeddings.shape[1]
        positions = torch.arange(total_seq_len, device=device).unsqueeze(0)
        position_embeddings = self.transformer.wpe(positions)
        
        # 4. 加入位置編碼
        x = combined_embeddings + position_embeddings
        
        # 5. 通過Transformer層
        all_attentions = []
        for block in self.transformer.h:
            x, attn_weights = block(x)
            all_attentions.append(attn_weights)
        x = self.transformer.ln_f(x)
        
        # 6. 預測下一個token
        logits = self.lm_head(x)
        
        # 7. 只返回文字部分的預測
        num_vis_tokens = visual_features.shape[1]
        text_logits = logits[:, num_vis_tokens:, :]
        
        return text_logits, all_attentions

class Block(nn.Module):
    def __init__(self, cfg, lora_rank):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg, lora_rank)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r=lora_rank)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', lora.Linear(4 * cfg.n_embd, cfg.n_embd, r=lora_rank))
        ]))

    def forward(self, x):
        attn_output, attn_weights = self.attn(self.ln_1(x))
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights

class Attention(nn.Module):
    def __init__(self, cfg, lora_rank):
        super().__init__()
        self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=lora_rank)
        self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r=lora_rank)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        mask = torch.tril(torch.ones(size, size))
        mask[:cfg.vtoken_size, :cfg.vtoken_size] = 1
        self.register_buffer('bias', mask.view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C)), att

class Config:
    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
        self.vtoken_size = 257



def create_attention_visualization(image, attention_weights, token, save_path=None):
    """
    Creates a visualization of attention weights for a single token and head.
    """
    # Create figure
    plt.figure(figsize=(5, 4))
    plt.subplots_adjust(top=0.9, bottom=0, left=0, right=1)
    
    # Reshape attention weights to match image patches
    if torch.is_tensor(attention_weights):
        image_attention = attention_weights.cpu().reshape(16, 16)


    image_cpu = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
    image_cpu = (image_cpu - image_cpu.min()) / (image_cpu.max() - image_cpu.min() + 1e-8)  # Normalize
    image_cpu = (image_cpu).astype(np.float32)

    # Create attention overlay
    plt.subplot(1, 1, 1)
    plt.imshow(image_cpu)
    
    # Resize attention map to match image size
    h, w = image_cpu.shape[1], image_cpu.shape[0]
    attention_resized = torch.nn.functional.interpolate(
        torch.tensor(image_attention, dtype=torch.float32)[None, None],
        size=(h, w),
        mode='bilinear',
        align_corners=False
    )[0, 0].numpy()
    
    attention_resized = (attention_resized - attention_resized.min()) / \
                        (attention_resized.max() - attention_resized.min() + 1e-8)
    
    # Apply attention overlay with jet colormap
    plt.imshow(attention_resized, alpha=0.5, cmap='jet', vmin=0, vmax=1)
    plt.title(f'{token}', y=0.98, pad=20,  fontsize=20, fontweight='bold', va='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()





class InferenceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # 收集所有圖片檔案
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
            
        # 返回圖片和檔名（不含副檔名）
        return image, os.path.splitext(filename)[0]

def load_trained_parameters(decoder, checkpoint_path):
    """載入訓練好的參數"""
    checkpoint = torch.load(checkpoint_path)
    decoder.load_state_dict(checkpoint['decoder_params'], strict=False)

    
    #print(sum(p.numel() for p in encoder.parameters() if p.requires_grad) + sum(p.numel() for p in decoder.parameters() if p.requires_grad))

def generate_captions_batch(encoder, decoder, tokenizer, dataloader, max_length=64, device='cuda'):
    """使用 DataLoader 批次生成圖片說明"""
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
            

            # 儲存每一步的輸出和attention
            generated_tokens = []
            attention_weights = []
            
            # 逐步生成
            for _ in range(max_length):
                logits, attn = decoder(visual_features, current_ids)
                next_token_logits = logits[:, -1, :]
                next_tokens = next_token_logits.argmax(dim=-1).unsqueeze(1)
                current_ids = torch.cat([current_ids, next_tokens], dim=1)

                generated_tokens.append(next_tokens)
                attention_weights.append(attn)

                #print(current_ids)
                
                # 檢查是否所有序列都生成了結束符
                #if all((current_ids == end_token).any(dim=1)):
                    #break
            
            # 解碼生成的序列
            for batch_idx, (filename, image) in enumerate(zip(batch_filenames, batch_images)):
                token_ids = current_ids[batch_idx]
                end_positions = (token_ids == end_token).nonzero(as_tuple=True)[0]
                if len(end_positions) > 1:  # 有找到第二個結束符
                    end_pos = end_positions[1].item()
                else:  # 沒有找到第二個結束符
                    end_pos = 30

                caption_ids = token_ids[:end_pos+1].tolist()  # 包含起始和結束token
                #print(caption_ids)
                caption = tokenizer.decode(caption_ids)

                output_dir = './attention_visualizations3'
                os.makedirs(output_dir, exist_ok=True)

                caption = caption[13:-14]
                words = caption.split()
                words = words +['.'] + ['<|endoftext|>']
                print(f"\nWords for {filename}:", words)

                # 為每個word創建attention map
                for hd in range(0, 12):
                    for token_idx, token in enumerate(words):
                        if token_idx >= len(attention_weights):
                            break
                            
                        curr_attn = attention_weights[token_idx][-1]  # 最後一層
                        head_attn = curr_attn[batch_idx, hd, 257+token_idx, 1:257]  # 使用batch_idx而不是ids
                        
                        save_path = os.path.join(
                            output_dir,
                            f"{hd}_{filename}_token_{token_idx:02d}_{token}.png"
                        )

                        #print(f"head_attn shape: {head_attn.shape}")
                        #print(f"head_attn min: {head_attn.min()}, max: {head_attn.max()}")
                        
                        create_attention_visualization(
                            image,
                            head_attn,
                            token,
                            save_path
                        )

                results[filename] = caption
                print(f"Generated caption for {filename}: {caption}")
    
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 1
    num_workers = 4
    
    # 載入 tokenizer
    tokenizer = BPETokenizer(
        encoder_file='encoder.json',
        vocab_file='vocab.bpe'
    )
    
    # 初始化模型
    encoder = ImageCaptioningEncoder().to(device)
    decoder_cfg = Config(checkpoint='./hw3_data/p2_data/decoder_model.bin')
    decoder = ModifiedDecoder(decoder_cfg, lora_rank=58).to(device)
    
    # 載入訓練好的參數
    load_trained_parameters(decoder, './sq10/best_model.pth') # ccc
    
    # 設定 transform
    model = timm.create_model('vit_large_patch14_clip_224.openai_ft_in1k', pretrained=True, num_classes=0)
    model = model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    
    transform_list = [
        timm.data.create_transform(**data_config, is_training=False)
    ]
    transform = transforms.Compose(transform_list)

    
    # 創建 dataset 和 dataloader
    test_dataset = InferenceDataset(
        image_dir='./hw3_data/p3_data/images',
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 生成說明文字
    results = generate_captions_batch(encoder, decoder, tokenizer, test_loader, device=device)
    
    # 儲存結果
    output_path = 'p32_result.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results have been saved to {output_path}")

if __name__ == '__main__':
    main()