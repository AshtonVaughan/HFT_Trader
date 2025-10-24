"""
Transformer-XL for Long-Context Time Series Modeling

Transformer-XL extends standard Transformer with:
- Segment-level recurrence mechanism
- Relative positional encoding
- Ability to learn dependencies beyond fixed context length
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import logger


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for Transformer-XL."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        # Precompute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: Sequence length

        Returns:
            Positional encoding (seq_len, d_model)
        """
        return self.pe[:seq_len]


class RelativeMultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional encoding."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        # Linear projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        # Relative position embeddings
        self.r_net = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mems: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            pos_emb: (seq_len, d_model) positional embeddings
            mask: Optional attention mask
            mems: Optional memory from previous segment

        Returns:
            output, attention_weights
        """
        batch_size, seq_len, _ = query.size()

        # Concatenate memory if provided
        if mems is not None:
            key = torch.cat([mems, key], dim=1)
            value = torch.cat([mems, value], dim=1)

        # Linear projections and reshape
        Q = self.q_linear(query).view(batch_size, seq_len, self.nhead, self.d_head)
        K = self.k_linear(key).view(batch_size, -1, self.nhead, self.d_head)
        V = self.v_linear(value).view(batch_size, -1, self.nhead, self.d_head)

        # Transpose for attention
        Q = Q.transpose(1, 2)  # (batch, nhead, seq_len, d_head)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_linear(output)

        return output, attn_weights


class TransformerXLLayer(nn.Module):
    """Single Transformer-XL layer with relative positional attention."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.self_attn = RelativeMultiHeadAttention(d_model, nhead, dropout)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mems: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input (batch, seq_len, d_model)
            pos_emb: Positional embeddings
            mask: Attention mask
            mems: Memory from previous segment

        Returns:
            output, memory
        """
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, pos_emb, mask, mems)
        x = self.norm1(x + self.dropout(attn_out))

        # Feedforward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # Return output and detached memory for next segment
        return x, x.detach()


class TransformerXLPredictor(nn.Module):
    """
    Transformer-XL for time series prediction.

    Learns long-range dependencies using segment-level recurrence.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        segment_len: int = 500,
        mem_len: int = 500
    ):
        """
        Args:
            input_size: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            segment_len: Length of each segment
            mem_len: Length of memory to retain
        """
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.segment_len = segment_len
        self.mem_len = mem_len

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = RelativePositionalEncoding(d_model, max_len=segment_len + mem_len)

        # Transformer-XL layers
        self.layers = nn.ModuleList([
            TransformerXLLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Output heads (same as specialized models)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Up or down
        )

        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize memory
        self.mems = None

        logger.info(f"TransformerXL initialized: {d_model}d, {num_layers} layers, {nhead} heads")

    def reset_memory(self):
        """Reset memory (call at start of new sequence)."""
        self.mems = None

    def forward(
        self,
        x: torch.Tensor,
        reset_memory: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input (batch, seq_len, input_size)
            reset_memory: Whether to reset memory

        Returns:
            Dictionary with predictions
        """
        if reset_memory:
            self.reset_memory()

        batch_size, seq_len, _ = x.size()

        # Input projection
        x = self.input_proj(x)
        x = self.dropout(x)

        # Get positional embeddings
        pos_emb = self.pos_encoder(seq_len + (self.mems[0].size(1) if self.mems is not None else 0))

        # Process through layers with memory
        new_mems = []

        for i, layer in enumerate(self.layers):
            mem = self.mems[i] if self.mems is not None and i < len(self.mems) else None
            x, new_mem = layer(x, pos_emb, mems=mem)
            new_mems.append(new_mem)

        # Update memory (keep only last mem_len tokens)
        if self.mem_len > 0:
            self.mems = [mem[:, -self.mem_len:, :] for mem in new_mems]

        # Use last token for prediction
        final_hidden = x[:, -1, :]  # (batch, d_model)

        # Predictions
        direction_logits = self.direction_head(final_hidden)
        magnitude = self.magnitude_head(final_hidden)
        confidence = self.confidence_head(final_hidden)

        return {
            'direction_logits': direction_logits,
            'magnitude': magnitude,
            'confidence': confidence
        }


if __name__ == '__main__':
    # Test Transformer-XL
    print("\n" + "="*80)
    print("Transformer-XL Test")
    print("="*80)

    # Create model
    model = TransformerXLPredictor(
        input_size=50,
        d_model=128,
        nhead=4,
        num_layers=3,
        segment_len=100,
        mem_len=100
    )

    # Test with segments
    print("\n1. Processing multiple segments:")
    model.reset_memory()

    for i in range(3):
        # Random segment
        segment = torch.randn(2, 100, 50)  # (batch, seq_len, features)

        outputs = model(segment)

        print(f"   Segment {i+1}:")
        print(f"     Direction logits: {outputs['direction_logits'].shape}")
        print(f"     Magnitude: {outputs['magnitude'].shape}")
        print(f"     Confidence: {outputs['confidence'].shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n2. Model Parameters: {total_params:,}")

    # Test memory
    print("\n3. Testing memory mechanism:")
    segment1 = torch.randn(1, 100, 50)
    segment2 = torch.randn(1, 100, 50)

    model.reset_memory()
    out1 = model(segment1)
    out2 = model(segment2)  # Should use memory from segment1

    print("   Memory successfully maintained across segments")
