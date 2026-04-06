"""
CLIP Text Encoder: 冻结参数，提取 token-level 和 sentence-level 特征。
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", max_length: int = 77):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, local_files_only=True)
        clip_model = CLIPModel.from_pretrained(model_name, local_files_only=True)
        self.text_model = clip_model.text_model
        self.text_projection = clip_model.text_projection
        for p in self.text_model.parameters():
            p.requires_grad = False
        for p in self.text_projection.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def tokenize(self, texts, device=None):
        tokens = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        if device is not None:
            tokens = {k: v.to(device) for k, v in tokens.items()}
        return tokens

    @torch.no_grad()
    def encode_tokens(self, texts, device=None):
        """
        返回 token-level 特征。
        Returns:
            token_features: (B, L, d)
            attention_mask: (B, L)
        """
        tokens = self.tokenize(texts, device=device)
        outputs = self.text_model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )
        hidden = outputs.last_hidden_state
        token_features = hidden @ self.text_projection.weight.T
        token_features = F.normalize(token_features, p=2, dim=-1)
        return token_features, tokens["attention_mask"]

    @torch.no_grad()
    def encode_sentence(self, texts, device=None):
        """
        返回 sentence-level [EOS] 特征。
        Returns:
            sentence_features: (B, d)
        """
        tokens = self.tokenize(texts, device=device)
        outputs = self.text_model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )
        pooled = outputs.pooler_output
        sentence_features = self.text_projection(pooled)
        sentence_features = F.normalize(sentence_features, p=2, dim=-1)
        return sentence_features
