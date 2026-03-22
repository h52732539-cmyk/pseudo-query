"""
CLIP Text Encoder 封装：冻结参数，提取 token-level 和 sentence-level 特征。
首次使用时自动从 HuggingFace 下载模型（支持镜像加速）。
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

# 如果未设置 HF_ENDPOINT，则使用国内镜像加速下载
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", max_length: int = 77):
        super().__init__()
        self.max_length = max_length
        print(f"[CLIPTextEncoder] Loading model: {model_name} (mirror: {os.environ.get('HF_ENDPOINT', 'default')}) ...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        clip_model = CLIPModel.from_pretrained(model_name)
        self.text_model = clip_model.text_model
        self.text_projection = clip_model.text_projection  # (hidden_dim, 512)
        # 冻结所有参数
        for p in self.text_model.parameters():
            p.requires_grad = False
        for p in self.text_projection.parameters():
            p.requires_grad = False
        print("[CLIPTextEncoder] Loaded and frozen.")

    @torch.no_grad()
    def tokenize(self, texts, device=None):
        """Tokenize 文本列表，返回 input_ids + attention_mask."""
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if device is not None:
            tokens = {k: v.to(device) for k, v in tokens.items()}
        return tokens

    @torch.no_grad()
    def encode_tokens(self, texts, device=None):
        """
        编码文本列表，返回 token-level 特征。
        Returns:
            token_features: (B, L, d)  — 投影后的 token 特征
            attention_mask: (B, L)
        """
        tokens = self.tokenize(texts, device=device)
        outputs = self.text_model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )
        # last_hidden_state: (B, L, hidden_dim)
        hidden = outputs.last_hidden_state
        # 投影到 CLIP 的共享空间
        token_features = hidden @ self.text_projection.weight.T  # (B, L, 512)
        # L2 归一化
        token_features = F.normalize(token_features, p=2, dim=-1)
        return token_features, tokens["attention_mask"]

    @torch.no_grad()
    def encode_sentence(self, texts, device=None):
        """
        编码文本列表，返回 sentence-level [EOS] 特征（CLIP 标准做法）。
        Returns:
            sentence_features: (B, d)
        """
        tokens = self.tokenize(texts, device=device)
        outputs = self.text_model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )
        # CLIP: pooled_output = last_hidden_state at EOS token position
        pooled = outputs.pooler_output  # (B, hidden_dim)
        sentence_features = self.text_projection(pooled)  # (B, 512)
        # L2 归一化
        sentence_features = F.normalize(sentence_features, p=2, dim=-1)
        return sentence_features
