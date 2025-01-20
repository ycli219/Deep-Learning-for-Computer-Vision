import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tokenizer import BPETokenizer
from p2_encoder import ImageCaptioningEncoder
from p2_decoder import ModifiedDecoder, Config
from tqdm import tqdm
import loralib as lora
import timm

class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, caption_file, tokenizer, max_length=64, modee='train'):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 載入資料
        with open(caption_file, 'r') as f:
            data = json.load(f)
        
        # 建立圖片ID到檔名的映射
        self.id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        
        # 整理資料
        self.samples = []
        for ann in data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            file_name = self.id_to_filename[image_id]
            self.samples.append({
                'image_id': image_id,
                'file_name': file_name,
                'caption': caption
            })
        
        model = timm.create_model('vit_large_patch14_clip_224.openai_ft_in1k', pretrained=True, num_classes=0)
        model = model.eval() # ccc ?
        data_config = timm.data.resolve_model_data_config(model)
        
        # 創建 transform
        if modee == 'train':
            transform_list = [
                transforms.TrivialAugmentWide(),
                timm.data.create_transform(**data_config, is_training=False)
            ]
        elif modee == 'val':
            transform_list = [
                timm.data.create_transform(**data_config, is_training=False)
            ]
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 載入圖片
        image_path = os.path.join(self.image_dir, sample['file_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # 處理文字
        end_token = self.tokenizer.encoder["<|endoftext|>"]
        caption_ids = self.tokenizer.encode(sample['caption'])
        
        # 如果序列太長，進行截斷
        if len(caption_ids) > self.max_length - 1:  # -1 是為了留空間給起始token
            caption_ids = caption_ids[:self.max_length - 1]
        
        # 準備輸入序列和標籤序列
        input_ids = [end_token] + caption_ids
        label_ids = caption_ids + [end_token]
        
        # 進行填充
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [end_token] * padding_length
            label_ids = label_ids + [-100] * padding_length
        
        return image, torch.tensor(input_ids), torch.tensor(label_ids)

def train_epoch(encoder, decoder, dataloader, optimizer, device, epoch, scheduler):
    encoder.train() 
    decoder.train()
    total_loss = 0
    
    # 使用 tqdm 包裝 dataloader
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, input_ids, label_ids) in enumerate(progress_bar):
        images = images.to(device)
        input_ids = input_ids.to(device)
        label_ids = label_ids.to(device)
        
        optimizer.zero_grad()
        
        visual_features = encoder(images)
        logits = decoder(visual_features, input_ids)

        #print(f"logits shape: {logits.shape}")
        #print(f"label_ids shape: {label_ids.shape}")

        # 修改這裡：先確保張量是連續的，再進行 view 操作
        logits = logits.contiguous()
        label_ids = label_ids.contiguous()
        
        loss = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing = 0.1)(
            logits.view(-1, logits.size(-1)),
            label_ids.view(-1)
        )
        
        loss.backward()
        optimizer.step()

        scheduler.step() # ccc
        
        total_loss += loss.item()
        
        # 更新進度條
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')
    
    return total_loss / len(dataloader)

def calculate_trainable_params(model):
    """計算模型中需要訓練的參數總數"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_trained_parameters(decoder, filename):
    """儲存 decoder 的所有訓練參數"""
    trained_params = {
        'decoder_params': {
            name: param.data
            for name, param in decoder.named_parameters()
            if param.requires_grad
        }
    }
    torch.save(trained_params, filename)

def validate(encoder, decoder, dataloader, device):
    encoder.eval()
    decoder.eval()
    total_loss = 0

    with torch.no_grad():
        # Using tqdm to wrap the dataloader for a progress bar
        for images, input_ids, label_ids in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)

            visual_features = encoder(images)
            logits = decoder(visual_features, input_ids)

            # Ensure tensors are contiguous before reshaping
            logits = logits.contiguous()
            label_ids = label_ids.contiguous()

            # Compute loss
            loss = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing = 0.1)(
                logits.view(-1, logits.size(-1)),
                label_ids.view(-1)
            )

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main():
    # 設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-4
    max_length = 64
    lora_rank = 58
    
    print(f"Using device: {device}")
    print(f"Max length: {max_length}")
    print(f"LoRA rank: {lora_rank}")
    
    # 載入 tokenizer
    tokenizer = BPETokenizer(
        encoder_file='encoder.json',
        vocab_file='vocab.bpe'
    )
    
    # 創建數據集
    train_dataset = ImageCaptionDataset(
        image_dir='./hw3_data/p2_data/images/train',
        caption_file='./hw3_data/p2_data/train.json',
        tokenizer=tokenizer,
        max_length=max_length,
        modee = 'train'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_dataset = ImageCaptionDataset(
        image_dir='./hw3_data/p2_data/images/val',
        caption_file='./hw3_data/p2_data/val.json',
        tokenizer=tokenizer,
        max_length=max_length,
        modee = 'val'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    encoder = ImageCaptioningEncoder().to(device)
    decoder_cfg = Config(checkpoint='./hw3_data/p2_data/decoder_model.bin')
    decoder = ModifiedDecoder(decoder_cfg, lora_rank=lora_rank).to(device)
    lora.mark_only_lora_as_trainable(decoder)

    # 將 projection layer 設為可訓練
    for param in decoder.projection.parameters():
        param.requires_grad = True
    
    # 計算總參數量
    encoder_params = calculate_trainable_params(encoder)
    decoder_params = calculate_trainable_params(decoder)
    total_params = encoder_params + decoder_params

    projection_params = sum(p.numel() for name, p in decoder.named_parameters() 
                          if p.requires_grad and 'projection' in name)
    lora_params = sum(p.numel() for name, p in decoder.named_parameters() 
                     if p.requires_grad and 'projection' not in name)
    
    print(f"\nParameter Statistics:")
    print(f"Encoder trainable parameters: {encoder_params:,}")
    print(f"Decoder projection layer parameters: {projection_params:,}")
    print(f"Decoder LoRA parameters: {lora_params:,}")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Remaining budget: {10_000_000 - total_params:,}")
    
    if total_params > 10_000_000:
        raise ValueError("Trainable parameters exceed 10M limit!")
    
    # 優化器 (包含所有要訓練的參數)
    trainable_params = [p for p in decoder.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-3, epochs = num_epochs, steps_per_epoch = len(train_loader))
    
    # 訓練循環
    best_loss = float('inf')
    
    for epoch in tqdm(range(num_epochs), desc='Training Progress'):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        train_loss = train_epoch(encoder, decoder, train_loader, optimizer, device, epoch+1, scheduler)
        print(f'Average loss: {train_loss:.4f}')

        val_loss = validate(encoder, decoder, val_loader, device)
        print(f'Validation loss: {val_loss:.4f}')
        
        # 儲存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            save_trained_parameters(
                decoder,
                './sq10/best_model.pth'
            )
        
        # 定期儲存檢查點
        if (epoch + 1) % 1 == 0:
            save_trained_parameters(
                decoder,
                f'./sq10/checkpoint_epoch_{epoch+1}.pth'
            )

if __name__ == '__main__':
    main()