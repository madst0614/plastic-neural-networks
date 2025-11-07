"""
Plastic Neural Networks (PNN) - Main Model Implementation

Paper: "Plastic Neural Networks: Learning Through Iterative Delta Refinement"
Author: Seungho Choi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class QueryKeyGate(nn.Module):
    """
    Query-Key Adaptive Gating Mechanism
    
    Computes element-wise compatibility between current representation
    and proposed delta to determine which dimensions to update.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Query: "What do I need?"
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        
        # Key: "What can I provide?"
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1) * math.sqrt(hidden_size))
        
    def forward(self, h: torch.Tensor, delta_raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Current representation [batch, seq_len, hidden]
            delta_raw: Proposed update [batch, seq_len, hidden]
            
        Returns:
            gate: Element-wise gates [batch, seq_len, hidden]
        """
        # Compute query and key
        query = self.query_proj(h)
        key = self.key_proj(delta_raw)
        
        # Element-wise compatibility
        compatibility = query * key
        
        # Temperature-scaled sigmoid
        gate = torch.sigmoid(compatibility / self.temperature)
        
        return gate


class DeltaRefiner(nn.Module):
    """
    Delta Refinement Module
    
    Single module applied recurrently to compute additive updates.
    Uses self-attention + FFN + adaptive gating.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_layer_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_layer_norm = nn.LayerNorm(hidden_size)
        
        # Adaptive gating
        self.gate = QueryKeyGate(hidden_size)
        
        # Zero-initialize final FFN layer for stable training
        nn.init.zeros_(self.ffn[3].weight)
        nn.init.zeros_(self.ffn[3].bias)
        
    def forward(
        self, 
        h: torch.Tensor, 
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute delta (additive update) for current representation.
        
        Args:
            h: Current representation [batch, seq_len, hidden]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            delta: Gated additive update [batch, seq_len, hidden]
        """
        # Self-attention with residual
        attn_out, _ = self.attention(h, h, h, key_padding_mask=attention_mask)
        h_attn = self.attn_layer_norm(h + self.attn_dropout(attn_out))
        
        # Feed-forward to compute raw delta
        delta_raw = self.ffn(h_attn)
        delta_raw = self.ffn_layer_norm(delta_raw)
        
        # Apply adaptive gating
        gate = self.gate(h, delta_raw)
        delta = gate * delta_raw
        
        return delta


class PlasticNeuralNetwork(nn.Module):
    """
    Plastic Neural Network (PNN)
    
    Learns through iterative delta refinement rather than deep layer stacks.
    Applies a single DeltaRefiner module recurrently.
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int = 2048,
        max_length: int = 128,
        num_steps: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_steps = num_steps
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        self.embedding_layer_norm = nn.LayerNorm(hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Single delta refiner (shared across steps)
        self.delta_refiner = DeltaRefiner(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            dropout=dropout
        )
        
        # MLM head
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following BERT"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_all_steps: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with recurrent delta refinement.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            return_all_steps: Return outputs from all refinement steps
            
        Returns:
            hidden: Final representation [batch, seq_len, hidden]
            or list of representations if return_all_steps=True
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # Add position embeddings
        position_ids = torch.arange(
            seq_len, 
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)
        
        hidden = token_embeds + position_embeds
        hidden = self.embedding_layer_norm(hidden)
        hidden = self.embedding_dropout(hidden)
        
        # Convert attention mask format (1 = valid, 0 = masked)
        attn_mask = None
        if attention_mask is not None:
            attn_mask = (attention_mask == 0)  # True = masked
        
        all_outputs = [hidden] if return_all_steps else None
        
        # Recurrent refinement
        for step in range(self.num_steps):
            delta = self.delta_refiner(hidden, attn_mask)
            hidden = hidden + delta
            
            if return_all_steps:
                all_outputs.append(hidden)
        
        if return_all_steps:
            return all_outputs
        else:
            return hidden
    
    def get_mlm_loss(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple:
        """
        Compute masked language modeling loss.
        
        Args:
            hidden: Hidden states [batch, seq_len, hidden]
            labels: Target labels [batch, seq_len]
            
        Returns:
            loss: Cross-entropy loss
            logits: Prediction logits [batch, seq_len, vocab]
        """
        logits = self.mlm_head(hidden)
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            ignore_index=-100
        )
        return loss, logits
    
    def compute_recurrent_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        step_weights: list = None
    ) -> tuple:
        """
        Compute weighted loss across all refinement steps.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels [batch, seq_len]
            step_weights: Weights for each step (default: [0.1, 0.2, 0.3, 0.4])
            
        Returns:
            total_loss: Weighted sum of step losses
            step_losses: Individual losses per step
        """
        if step_weights is None:
            step_weights = [0.1, 0.2, 0.3, 0.4]
        
        # Get outputs from all steps
        all_outputs = self.forward(
            input_ids, 
            attention_mask, 
            return_all_steps=True
        )
        
        total_loss = 0.0
        step_losses = []
        
        # Skip embedding (step 0), compute loss for refinement steps (1-4)
        for step_idx, hidden in enumerate(all_outputs[1:], start=0):
            loss, _ = self.get_mlm_loss(hidden, labels)
            weight = step_weights[step_idx]
            total_loss += weight * loss
            step_losses.append(loss.item())
        
        return total_loss, step_losses


def create_pnn_model(config: dict = None) -> PlasticNeuralNetwork:
    """
    Factory function to create PNN model with config.
    
    Args:
        config: Model configuration dict
        
    Returns:
        model: PlasticNeuralNetwork instance
    """
    if config is None:
        # Default config from paper
        config = {
            'vocab_size': 30522,
            'hidden_size': 768,
            'num_heads': 12,
            'intermediate_size': 2048,
            'max_length': 128,
            'num_steps': 4,
            'dropout': 0.1
        }
    
    model = PlasticNeuralNetwork(**config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Created PNN with {total_params:,} parameters ({total_params/1e6:.1f}M)")
    
    return model


if __name__ == "__main__":
    # Example usage
    model = create_pnn_model()
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = input_ids.clone()
    
    # Mask some tokens
    mask_indices = torch.rand(batch_size, seq_len) < 0.15
    labels[~mask_indices] = -100
    input_ids[mask_indices] = 103  # [MASK] token
    
    # Forward pass
    hidden = model(input_ids, attention_mask)
    print(f"Output shape: {hidden.shape}")
    
    # Compute loss
    loss, logits = model.get_mlm_loss(hidden, labels)
    print(f"Loss: {loss.item():.4f}")
    
    # Recurrent loss (all steps)
    total_loss, step_losses = model.compute_recurrent_loss(
        input_ids, attention_mask, labels
    )
    print(f"Step losses: {[f'{l:.4f}' for l in step_losses]}")
    print(f"Total weighted loss: {total_loss.item():.4f}")
