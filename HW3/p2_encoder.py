import torch
import torch.nn as nn
import timm

class ImageCaptioningEncoder(nn.Module):
    def __init__(self, 
                 vision_model_name="vit_large_patch14_clip_224.openai_ft_in1k"):  # 使用 CLIP 預訓練的 ViT-L
        super().__init__()
        
        # 載入 timm 視覺編碼器
        self.vision_encoder = timm.create_model(
            vision_model_name,
            pretrained=True,
            num_classes=0  # 移除分類頭
        )
        
        # Freeze vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: 圖片張量，shape [batch_size, 3, 224, 224]
        Returns:
            projected_features: shape [batch_size, num_patches, decoder_hidden_size]
        """
        # Get vision features
        vision_features = self.vision_encoder.forward_features(pixel_values)  
        # vision_features shape: [batch_size, num_patches, vit_hidden_size]
        
        return vision_features