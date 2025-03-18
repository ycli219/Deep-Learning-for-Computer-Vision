import math
import collections
import torch
from torch import nn
import torch.nn.functional as F
import loralib as lora

class ModifiedDecoder(nn.Module):
    def __init__(self, cfg, lora_rank=58):
        super().__init__()
        self.cfg = cfg

        # Projection layer
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
        x = self.transformer.h(x)
        x = self.transformer.ln_f(x)
        
        # 6. 預測下一個token
        logits = self.lm_head(x)
        
        # 7. 只返回文字部分的預測
        num_vis_tokens = visual_features.shape[1]
        text_logits = logits[:, num_vis_tokens:, :]
        
        return text_logits

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
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

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
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class Config:
    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
        self.vtoken_size = 257