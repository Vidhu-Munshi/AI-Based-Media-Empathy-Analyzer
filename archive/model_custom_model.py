"""
Custom Multi-Task Empathy Analyzer Model
========================================
Architecture: BiLSTM + Self-Attention Encoder with 4 Task-Specific Heads
- Emotion Head: Multi-label classification (20 classes)
- Empathy Head: Regression ([-3, +3])
- Style Head: Single-class classification (8 classes)
- IMDb Rating Head: Regression ([1, 10]) — movie-level, uses aggregated character features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the custom empathy analyzer model."""
    # Vocabulary / Embedding
    vocab_size: int = 12000
    embed_dim: int = 256
    max_seq_len: int = 256
    pad_token_id: int = 0
    unk_token_id: int = 1

    # Encoder (BiLSTM + Attention)
    encoder_type: str = "bilstm_attention"  # or "transformer"
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    attention_heads: int = 4
    attention_dim: int = 256

    # Task Heads
    num_emotions: int = 20
    num_styles: int = 8
    empathy_range: Tuple[float, float] = (-3.0, 3.0)
    imdb_range: Tuple[float, float] = (1.0, 10.0)

    # Loss Weights (multi-task balancing)
    lambda_emotion: float = 1.0
    lambda_empathy: float = 1.0
    lambda_style: float = 1.0
    lambda_imdb: float = 0.5  # lower since movie-level, fewer samples

    # Character Aggregation
    agg_method: str = "attention"  # "mean", "max", "attention"


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for sequence encoding."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, seq_len, embed_dim)
        mask: (batch, seq_len) — 1 for valid, 0 for padding
        """
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            # mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (batch, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(out)


class BiLSTMAttentionEncoder(nn.Module):
    """BiLSTM encoder with self-attention pooling."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embed_dim,
            padding_idx=config.pad_token_id
        )

        lstm_hidden = config.hidden_dim // 2 if config.bidirectional else config.hidden_dim
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=lstm_hidden,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )

        self.attention = MultiHeadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.attention_heads,
            dropout=config.dropout
        )

        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns: (batch, seq_len, hidden_dim) — contextualized token representations
        """
        # Embedding
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        x = self.dropout(x)

        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # Self-attention
        attn_out = self.attention(lstm_out, attention_mask)

        # Residual + LayerNorm
        out = self.layer_norm(lstm_out + attn_out)
        return out


class TransformerEncoder(nn.Module):
    """Alternative: Pure Transformer encoder (like BERT but smaller)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embed_dim,
            padding_idx=config.pad_token_id
        )
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)

        x = self.embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)

        # Transformer expects mask as (batch, seq_len) with True for padding
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return self.layer_norm(out)


class EmotionHead(nn.Module):
    """Multi-label emotion classification head (20 classes)."""

    def __init__(self, hidden_dim: int, num_emotions: int, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_dim) or (batch, hidden_dim) if pooled
        Returns: (batch, num_emotions) — logits for each emotion
        """
        if x.dim() == 3:
            # Mean pooling over sequence
            x = x.mean(dim=1)
        return self.classifier(x)  # (batch, num_emotions)


class EmpathyHead(nn.Module):
    """Regression head for empathy score [-3, +3]."""

    def __init__(self, hidden_dim: int, dropout: float = 0.3, out_range: Tuple[float, float] = (-3.0, 3.0)):
        super().__init__()
        self.out_min, self.out_max = out_range
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Output in [-1, 1], will scale to [out_min, out_max]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        raw = self.regressor(x).squeeze(-1)  # (batch,) in [-1, 1]
        # Scale to target range
        return self.out_min + (raw + 1) * (self.out_max - self.out_min) / 2


class StyleHead(nn.Module):
    """Single-class communication style classification (8 classes)."""

    def __init__(self, hidden_dim: int, num_styles: int, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_styles)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.classifier(x)  # (batch, num_styles)


class CharacterAggregator(nn.Module):
    """Aggregates dialogue-level features to character-level profiles."""

    def __init__(self, hidden_dim: int, method: str = "attention"):
        super().__init__()
        self.method = method
        self.hidden_dim = hidden_dim

        if method == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )

    def forward(
        self,
        dialogue_features: torch.Tensor,  # (num_dialogues, hidden_dim)
        character_ids: torch.Tensor,      # (num_dialogues,) — character index per dialogue
        num_characters: int
    ) -> torch.Tensor:
        """
        Returns: (num_characters, hidden_dim) — aggregated character representations
        """
        if self.method == "mean":
            # Simple mean pooling per character
            char_features = torch.zeros(num_characters, self.hidden_dim, device=dialogue_features.device)
            char_counts = torch.zeros(num_characters, device=dialogue_features.device)
            char_features.index_add_(0, character_ids, dialogue_features)
            char_counts.index_add_(0, character_ids, torch.ones_like(character_ids, dtype=torch.float))
            char_features = char_features / char_counts.unsqueeze(-1).clamp(min=1)
            return char_features

        elif self.method == "max":
            # Max pooling per character
            char_features = torch.full((num_characters, self.hidden_dim), -1e9, device=dialogue_features.device)
            for i in range(num_characters):
                mask = (character_ids == i)
                if mask.any():
                    char_features[i] = dialogue_features[mask].max(dim=0).values
            return char_features

        elif self.method == "attention":
            # Attention-weighted pooling
            char_features = torch.zeros(num_characters, self.hidden_dim, device=dialogue_features.device)
            for i in range(num_characters):
                mask = (character_ids == i)
                if mask.any():
                    char_dial = dialogue_features[mask]  # (n_char_dials, hidden_dim)
                    weights = self.attention(char_dial)  # (n_char_dials, 1)
                    weights = F.softmax(weights, dim=0)
                    char_features[i] = (char_dial * weights).sum(dim=0)
            return char_features

        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")


class IMDbRatingHead(nn.Module):
    """Movie-level IMDb rating prediction from aggregated character profiles."""

    def __init__(self, hidden_dim: int, num_characters: int, dropout: float = 0.3, out_range: Tuple[float, float] = (1.0, 10.0)):
        super().__init__()
        self.out_min, self.out_max = out_range
        # Input: concatenated character profiles (num_characters * hidden_dim)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * num_characters, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1], scale to [1, 10]
        )

    def forward(self, char_profiles: torch.Tensor) -> torch.Tensor:
        """
        char_profiles: (batch, num_characters, hidden_dim) or (num_characters, hidden_dim)
        Returns: (batch,) or scalar — predicted IMDb rating
        """
        if char_profiles.dim() == 2:
            char_profiles = char_profiles.unsqueeze(0)  # (1, num_chars, hidden_dim)

        batch, num_chars, hidden_dim = char_profiles.shape
        x = char_profiles.view(batch, -1)  # (batch, num_chars * hidden_dim)
        raw = self.regressor(x).squeeze(-1)  # (batch,) in [0, 1]
        return self.out_min + raw * (self.out_max - self.out_min)


class EmpathyAnalyzerModel(nn.Module):
    """
    Main model: Shared encoder + 4 task heads + character aggregation + IMDb head.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Shared Encoder
        if config.encoder_type == "transformer":
            self.encoder = TransformerEncoder(config)
            encoder_out_dim = config.embed_dim
        else:
            self.encoder = BiLSTMAttentionEncoder(config)
            encoder_out_dim = config.hidden_dim

        # Dialogue-Level Heads
        self.emotion_head = EmotionHead(encoder_out_dim, config.num_emotions, config.dropout)
        self.empathy_head = EmpathyHead(encoder_out_dim, config.dropout, config.empathy_range)
        self.style_head = StyleHead(encoder_out_dim, config.num_styles, config.dropout)

        # Character Aggregation
        self.char_aggregator = CharacterAggregator(encoder_out_dim, config.agg_method)

        # Movie-Level IMDb Head
        # Note: num_characters varies per movie; we'll use a fixed max and mask
        self.max_characters = 20  # Max characters per movie
        self.imdb_head = IMDbRatingHead(encoder_out_dim, self.max_characters, config.dropout, config.imdb_range)

    def forward(
        self,
        input_ids: torch.Tensor,           # (batch, seq_len) — batch of dialogue lines
        attention_mask: Optional[torch.Tensor] = None,  # (batch, seq_len)
        character_ids: Optional[torch.Tensor] = None,   # (batch,) — character index per dialogue
        num_characters: Optional[int] = None,           # Total unique characters in this movie
        return_char_profiles: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for dialogue-level tasks.
        For IMDb rating, call `predict_imdb` separately with aggregated profiles.
        """
        # Shared encoding
        encoded = self.encoder(input_ids, attention_mask)  # (batch, seq_len, hidden_dim)

        # Dialogue-level predictions
        emotion_logits = self.emotion_head(encoded)        # (batch, num_emotions)
        empathy_scores = self.empathy_head(encoded)        # (batch,)
        style_logits = self.style_head(encoded)            # (batch, num_styles)

        outputs = {
            "emotion_logits": emotion_logits,
            "empathy_scores": empathy_scores,
            "style_logits": style_logits,
            "encoded": encoded
        }

        # Character aggregation (if character_ids provided)
        if character_ids is not None and num_characters is not None:
            char_profiles = self.char_aggregator(encoded.mean(dim=1), character_ids, num_characters)
            outputs["character_profiles"] = char_profiles  # (num_characters, hidden_dim)

            if return_char_profiles:
                return outputs

        return outputs

    def predict_imdb(self, character_profiles: torch.Tensor) -> torch.Tensor:
        """
        Predict IMDb rating from aggregated character profiles.
        character_profiles: (num_characters, hidden_dim) or (batch, num_characters, hidden_dim)
        """
        # Pad or truncate to max_characters
        if character_profiles.dim() == 2:
            character_profiles = character_profiles.unsqueeze(0)  # (1, num_chars, hidden_dim)

        batch, num_chars, hidden_dim = character_profiles.shape

        if num_chars > self.max_characters:
            character_profiles = character_profiles[:, :self.max_characters]
        elif num_chars < self.max_characters:
            padding = torch.zeros(batch, self.max_characters - num_chars, hidden_dim,
                                device=character_profiles.device)
            character_profiles = torch.cat([character_profiles, padding], dim=1)

        return self.imdb_head(character_profiles)

    def get_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        character_profiles: Optional[torch.Tensor] = None,
        imdb_target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.
        targets dict keys: emotion_labels (multi-hot), empathy_scores, style_labels, imdb_rating (optional)
        """
        losses = {}
        total_loss = 0.0

        # Emotion: Binary Cross Entropy with Logits (multi-label)
        if "emotion_labels" in targets:
            emotion_loss = F.binary_cross_entropy_with_logits(
                outputs["emotion_logits"], targets["emotion_labels"].float()
            )
            losses["emotion"] = emotion_loss.item()
            total_loss += self.config.lambda_emotion * emotion_loss

        # Empathy: MSE Loss
        if "empathy_scores" in targets:
            empathy_loss = F.mse_loss(outputs["empathy_scores"], targets["empathy_scores"].float())
            losses["empathy"] = empathy_loss.item()
            total_loss += self.config.lambda_empathy * empathy_loss

        # Style: Cross Entropy (single-class)
        if "style_labels" in targets:
            style_loss = F.cross_entropy(outputs["style_logits"], targets["style_labels"].long())
            losses["style"] = style_loss.item()
            total_loss += self.config.lambda_style * style_loss

        # IMDb: MSE Loss (if character profiles and target provided)
        if imdb_target is not None and character_profiles is not None:
            imdb_pred = self.predict_imdb(character_profiles)
            imdb_loss = F.mse_loss(imdb_pred, imdb_target.float())
            losses["imdb"] = imdb_loss.item()
            total_loss += self.config.lambda_imdb * imdb_loss

        losses["total"] = total_loss.item()
        return total_loss, losses


def create_model(config: Optional[ModelConfig] = None) -> EmpathyAnalyzerModel:
    """Factory function to create model with default or custom config."""
    if config is None:
        config = ModelConfig()
    return EmpathyAnalyzerModel(config)


# Emotion & Style label mappings (for reference)
EMOTION_LABELS = [
    "anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral",
    "anticipation", "trust", "contempt", "embarrassment", "pride", "gratitude",
    "empathy", "hostility", "condescension", "supportiveness", "dismissiveness",
    "mocking", "respectful"
]

STYLE_LABELS = [
    "monologuer", "inquisitive", "hesitant", "direct",
    "repetitive", "diverse", "emotional", "analytical"
]

EMOTION_TO_ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
STYLE_TO_ID = {label: i for i, label in enumerate(STYLE_LABELS)}
ID_TO_EMOTION = {i: label for i, label in enumerate(EMOTION_LABELS)}
ID_TO_STYLE = {i: label for i, label in enumerate(STYLE_LABELS)}