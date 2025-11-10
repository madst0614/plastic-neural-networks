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


class DeltaRefinerExpanded(nn.Module):
    """
    Experiment 1: Delta Refiner with Dual Attention + Dual FFN

    Structure: Attention1 → FFN1 → Attention2 → FFN2 → Gate
    Increases capacity by stacking two transformer-like blocks within a single refiner.
    Total added params: ~5.55M (attention: 2.4M, FFN: 3.15M)
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

        # First attention block
        self.attention1 = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn1_layer_norm = nn.LayerNorm(hidden_size)
        self.attn1_dropout = nn.Dropout(dropout)

        # First FFN block
        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn1_layer_norm = nn.LayerNorm(hidden_size)

        # Second attention block
        self.attention2 = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn2_layer_norm = nn.LayerNorm(hidden_size)
        self.attn2_dropout = nn.Dropout(dropout)

        # Second FFN block
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn2_layer_norm = nn.LayerNorm(hidden_size)

        # Adaptive gating
        self.gate = QueryKeyGate(hidden_size)

        # Zero-initialize final FFN layers for stable training
        nn.init.zeros_(self.ffn1[3].weight)
        nn.init.zeros_(self.ffn1[3].bias)
        nn.init.zeros_(self.ffn2[3].weight)
        nn.init.zeros_(self.ffn2[3].bias)

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
        # First transformer block: Attention1 + FFN1
        attn_out1, _ = self.attention1(h, h, h, key_padding_mask=attention_mask)
        h_attn1 = self.attn1_layer_norm(h + self.attn1_dropout(attn_out1))

        ffn_out1 = self.ffn1(h_attn1)
        h_ffn1 = self.ffn1_layer_norm(h_attn1 + ffn_out1)

        # Second transformer block: Attention2 + FFN2
        attn_out2, _ = self.attention2(h_ffn1, h_ffn1, h_ffn1, key_padding_mask=attention_mask)
        h_attn2 = self.attn2_layer_norm(h_ffn1 + self.attn2_dropout(attn_out2))

        delta_raw = self.ffn2(h_attn2)
        delta_raw = self.ffn2_layer_norm(delta_raw)

        # Apply adaptive gating
        gate = self.gate(h, delta_raw)
        delta = gate * delta_raw

        return delta


class PlasticNeuralNetworkExp1(nn.Module):
    """
    Experiment 1: PNN with Dual Attention + Dual FFN Refiner

    Each refiner contains 2 transformer blocks (Attention + FFN) stacked sequentially.
    This increases capacity while maintaining the recurrent refinement paradigm.
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

        # Embeddings (same as original)
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        self.embedding_layer_norm = nn.LayerNorm(hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        # Single expanded delta refiner (shared across steps)
        self.delta_refiner = DeltaRefinerExpanded(
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
        """Forward pass with recurrent delta refinement."""
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

        # Convert attention mask format
        attn_mask = None
        if attention_mask is not None:
            attn_mask = (attention_mask == 0)

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

    def get_mlm_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> tuple:
        """Compute masked language modeling loss."""
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
        step_weights: list = None,
        return_accuracies: bool = False
    ) -> tuple:
        """Compute weighted loss across all refinement steps.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels [batch, seq_len]
            step_weights: Weights for each step (default: [0.1, 0.2, 0.3, 0.4])
            return_accuracies: If True, also return step-wise accuracies

        Returns:
            total_loss: Weighted sum of step losses
            step_losses: Individual losses per step
            step_accs: (optional) Individual accuracies per step
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
        step_accs = [] if return_accuracies else None

        # Skip embedding (step 0), compute loss for refinement steps (1-4)
        for step_idx, hidden in enumerate(all_outputs[1:], start=0):
            loss, logits = self.get_mlm_loss(hidden, labels)
            weight = step_weights[step_idx]
            total_loss += weight * loss
            step_losses.append(loss.item())

            if return_accuracies:
                # Calculate accuracy for this step (detach to save memory)
                with torch.no_grad():
                    preds = logits.detach().argmax(dim=-1).view(-1)  # [B*L]
                    labels_flat = labels.view(-1)  # [B*L]
                    mask = (labels_flat != -100)  # [B*L]
                    correct = ((preds == labels_flat) & mask).sum().item()
                    total_tokens = mask.sum().item()
                    acc = correct / total_tokens if total_tokens > 0 else 0.0
                    step_accs.append(acc)

            # Delete logits immediately to free memory (~7.5GB per step)
            del logits

        if return_accuracies:
            return total_loss, step_losses, step_accs
        else:
            return total_loss, step_losses


class PlasticNeuralNetworkExp2(nn.Module):
    """
    Experiment 2: PNN with Dual Refiners (Alternating)

    Uses two independent refiners applied alternately for increased capacity.
    Each iteration applies refiner1 then refiner2.

    Note: Step outputs order: [emb, r1, r2, r1, r2] for 2 iterations
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int = 2048,
        max_length: int = 128,
        num_iterations: int = 4,  # 4 iterations × 2 refiners = 8 steps
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_iterations = num_iterations
        self.num_steps = num_iterations * 2  # For compatibility

        # Embeddings (same as original)
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        self.embedding_layer_norm = nn.LayerNorm(hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        # Two independent delta refiners
        self.delta_refiner1 = DeltaRefiner(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            dropout=dropout
        )

        self.delta_refiner2 = DeltaRefiner(
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
        """Forward pass with alternating dual refiners."""
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

        # Convert attention mask format
        attn_mask = None
        if attention_mask is not None:
            attn_mask = (attention_mask == 0)

        all_outputs = [hidden] if return_all_steps else None

        # Alternating dual refinement
        for iteration in range(self.num_iterations):
            # Apply refiner 1
            delta1 = self.delta_refiner1(hidden, attn_mask)
            hidden = hidden + delta1

            if return_all_steps:
                all_outputs.append(hidden)

            # Apply refiner 2
            delta2 = self.delta_refiner2(hidden, attn_mask)
            hidden = hidden + delta2

            if return_all_steps:
                all_outputs.append(hidden)

        if return_all_steps:
            return all_outputs
        else:
            return hidden

    def get_mlm_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> tuple:
        """Compute masked language modeling loss."""
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
        step_weights: list = None,
        return_accuracies: bool = False
    ) -> tuple:
        """Compute weighted loss across all refinement steps.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels [batch, seq_len]
            step_weights: Weights for each step (default: [0.1, 0.2, 0.3, 0.4])
            return_accuracies: If True, also return step-wise accuracies

        Returns:
            total_loss: Weighted sum of step losses
            step_losses: Individual losses per step
            step_accs: (optional) Individual accuracies per step
        """
        if step_weights is None:
            step_weights = [0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.23]  # 8 steps total

        # Get outputs from all steps
        all_outputs = self.forward(
            input_ids,
            attention_mask,
            return_all_steps=True
        )

        total_loss = 0.0
        step_losses = []
        step_accs = [] if return_accuracies else None

        # Skip embedding (step 0), compute loss for refinement steps (1-8)
        for step_idx, hidden in enumerate(all_outputs[1:], start=0):
            loss, logits = self.get_mlm_loss(hidden, labels)
            weight = step_weights[step_idx]
            total_loss += weight * loss
            step_losses.append(loss.item())

            if return_accuracies:
                # Calculate accuracy for this step (detach to save memory)
                with torch.no_grad():
                    preds = logits.detach().argmax(dim=-1).view(-1)  # [B*L]
                    labels_flat = labels.view(-1)  # [B*L]
                    mask = (labels_flat != -100)  # [B*L]
                    correct = ((preds == labels_flat) & mask).sum().item()
                    total_tokens = mask.sum().item()
                    acc = correct / total_tokens if total_tokens > 0 else 0.0
                    step_accs.append(acc)

            # Delete logits immediately to free memory (~7.5GB per step)
            del logits

        if return_accuracies:
            return total_loss, step_losses, step_accs
        else:
            return total_loss, step_losses


class DeltaRefinerExtendedDepth(nn.Module):
    """
    Experiment 4: Delta Refiner with Extended Depth (3 Transformer Blocks)

    Structure: (Attention1 → FFN1) → (Attention2 → FFN2) → (Attention3 → FFN3) → Gate
    Tests if deeper structure (3 blocks) outperforms dual blocks (Exp1).
    Each FFN: 768 → 2048 → 768 (proven size from Exp1)

    Total params per refiner: ~16.7M
    Applied recurrently for 4 steps = 12 transformer passes total
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int = 2048,
        dropout: float = 0.1,
        num_blocks: int = 3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks

        # Create transformer blocks dynamically
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = nn.ModuleDict({
                'attention': nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'attn_layer_norm': nn.LayerNorm(hidden_size),
                'attn_dropout': nn.Dropout(dropout),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_size, intermediate_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(intermediate_size, hidden_size),
                    nn.Dropout(dropout)
                ),
                'ffn_layer_norm': nn.LayerNorm(hidden_size)
            })
            # Zero-initialize final FFN layer for stable training
            nn.init.zeros_(block['ffn'][3].weight)
            nn.init.zeros_(block['ffn'][3].bias)
            self.blocks.append(block)

        # Adaptive gating
        self.gate = QueryKeyGate(hidden_size)

    def forward(
        self,
        h: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute delta through extended depth processing.

        Args:
            h: Current representation [batch, seq_len, hidden]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            delta: Gated additive update [batch, seq_len, hidden]
        """
        h_current = h

        # Process through all transformer blocks
        for i, block in enumerate(self.blocks):
            # Attention block
            attn_out, _ = block['attention'](
                h_current, h_current, h_current,
                key_padding_mask=attention_mask
            )
            h_attn = block['attn_layer_norm'](
                h_current + block['attn_dropout'](attn_out)
            )

            # FFN block
            ffn_out = block['ffn'](h_attn)
            h_current = block['ffn_layer_norm'](h_attn + ffn_out)

        # Final output becomes raw delta
        delta_raw = h_current

        # Apply adaptive gating (compare with original h, not h_current)
        gate = self.gate(h, delta_raw)
        delta = gate * delta_raw

        return delta


class PlasticNeuralNetworkExp4(nn.Module):
    """
    Experiment 4: PNN with Extended Depth (3 Transformer Blocks per Refiner)

    Tests if deeper structure improves over Exp1's dual blocks.
    Structure: 3 blocks × 4 steps = 12 transformer passes

    Comparison:
    - Exp1: 2 blocks × 4 steps = 8 passes (~59M params)
    - Exp4: 3 blocks × 4 steps = 12 passes (~75M params)

    Each block: Attention + FFN (768→2048→768)
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int = 2048,
        max_length: int = 128,
        num_steps: int = 4,
        dropout: float = 0.1,
        num_blocks: int = 3
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

        # Extended depth delta refiner (shared across steps)
        self.delta_refiner = DeltaRefinerExtendedDepth(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
            num_blocks=num_blocks
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
        """Forward pass with recurrent delta refinement."""
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_ids = torch.arange(
            seq_len,
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)

        hidden = token_embeds + position_embeds
        hidden = self.embedding_layer_norm(hidden)
        hidden = self.embedding_dropout(hidden)

        # Convert attention mask format
        attn_mask = None
        if attention_mask is not None:
            attn_mask = (attention_mask == 0)

        all_outputs = [hidden] if return_all_steps else None

        # Recurrent refinement with extended depth
        for step in range(self.num_steps):
            delta = self.delta_refiner(hidden, attn_mask)
            hidden = hidden + delta

            if return_all_steps:
                all_outputs.append(hidden)

        if return_all_steps:
            return all_outputs
        else:
            return hidden

    def get_mlm_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> tuple:
        """Compute masked language modeling loss."""
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
        step_weights: list = None,
        return_accuracies: bool = False
    ) -> tuple:
        """Compute weighted loss across all refinement steps."""
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
        step_accs = [] if return_accuracies else None

        # Skip embedding (step 0), compute loss for refinement steps (1-4)
        for step_idx, hidden in enumerate(all_outputs[1:], start=0):
            loss, logits = self.get_mlm_loss(hidden, labels)
            weight = step_weights[step_idx]
            total_loss += weight * loss
            step_losses.append(loss.item())

            if return_accuracies:
                # Calculate accuracy for this step
                with torch.no_grad():
                    preds = logits.detach().argmax(dim=-1).view(-1)  # [B*L]
                    labels_flat = labels.view(-1)  # [B*L]
                    mask = (labels_flat != -100)  # [B*L]
                    correct = ((preds == labels_flat) & mask).sum().item()
                    total_tokens = mask.sum().item()
                    acc = correct / total_tokens if total_tokens > 0 else 0.0
                    step_accs.append(acc)

            # Delete logits immediately to free memory
            del logits

        if return_accuracies:
            return total_loss, step_losses, step_accs
        else:
            return total_loss, step_losses


class PlasticNeuralNetworkExp5(nn.Module):
    """
    Experiment 5: PNN with 11 Transformer Blocks (BERT-baseline parameter matched)

    Tests structural efficiency by matching BERT-base's ~110M parameters.
    Structure: 11 blocks × 4 steps = 44 transformer passes

    Comparison:
    - BERT-base: 12 layers, ~110M params, 12 passes
    - Exp5: 11 blocks × 4 steps, ~108M params, 44 passes

    Key insight: Same params, but 3.7x more passes through recurrent reuse.
    Tests if "parameter reuse + depth" beats "unique layers"

    Each block: Attention + FFN (768→2048→768)
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int = 2048,
        max_length: int = 128,
        num_steps: int = 4,
        dropout: float = 0.1,
        num_blocks: int = 11
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

        # Extended depth delta refiner with 11 blocks (shared across steps)
        self.delta_refiner = DeltaRefinerExtendedDepth(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
            num_blocks=num_blocks
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
        """Forward pass with recurrent delta refinement."""
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_ids = torch.arange(
            seq_len,
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)

        hidden = token_embeds + position_embeds
        hidden = self.embedding_layer_norm(hidden)
        hidden = self.embedding_dropout(hidden)

        # Convert attention mask format
        attn_mask = None
        if attention_mask is not None:
            attn_mask = (attention_mask == 0)

        all_outputs = [hidden] if return_all_steps else None

        # Recurrent refinement with 11 blocks
        for step in range(self.num_steps):
            delta = self.delta_refiner(hidden, attn_mask)
            hidden = hidden + delta

            if return_all_steps:
                all_outputs.append(hidden)

        if return_all_steps:
            return all_outputs
        else:
            return hidden

    def get_mlm_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> tuple:
        """Compute masked language modeling loss."""
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
        step_weights: list = None,
        return_accuracies: bool = False
    ) -> tuple:
        """Compute weighted loss across all refinement steps."""
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
        step_accs = [] if return_accuracies else None

        # Skip embedding (step 0), compute loss for refinement steps (1-4)
        for step_idx, hidden in enumerate(all_outputs[1:], start=0):
            loss, logits = self.get_mlm_loss(hidden, labels)
            weight = step_weights[step_idx]
            total_loss += weight * loss
            step_losses.append(loss.item())

            if return_accuracies:
                # Calculate accuracy for this step
                with torch.no_grad():
                    preds = logits.detach().argmax(dim=-1).view(-1)  # [B*L]
                    labels_flat = labels.view(-1)  # [B*L]
                    mask = (labels_flat != -100)  # [B*L]
                    correct = ((preds == labels_flat) & mask).sum().item()
                    total_tokens = mask.sum().item()
                    acc = correct / total_tokens if total_tokens > 0 else 0.0
                    step_accs.append(acc)

            # Delete logits immediately to free memory
            del logits

        if return_accuracies:
            return total_loss, step_losses, step_accs
        else:
            return total_loss, step_losses


class PlasticNeuralNetworkExp3(nn.Module):
    """
    Experiment 3: PNN with Big Single FFN

    Tests whether capacity alone (big FFN) or structure (dual blocks) is more important.
    Uses single attention + big FFN (intermediate_size=4096) to match Exp1's parameter count.

    Comparison:
    - Exp1: 2 attention + 2 FFN (2048 each) = ~11M params/refiner
    - Exp3: 1 attention + 1 FFN (4096) = ~8.6M params/refiner

    Both applied recurrently for 4 steps.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int = 4096,  # Bigger than default!
        max_length: int = 128,
        num_steps: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_steps = num_steps

        # Embeddings (same as original)
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        self.embedding_layer_norm = nn.LayerNorm(hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        # Single big delta refiner (shared across steps)
        self.delta_refiner = DeltaRefiner(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,  # 4096!
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
        """Forward pass with recurrent delta refinement."""
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_ids = torch.arange(
            seq_len,
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)

        hidden = token_embeds + position_embeds
        hidden = self.embedding_layer_norm(hidden)
        hidden = self.embedding_dropout(hidden)

        # Convert attention mask format
        attn_mask = None
        if attention_mask is not None:
            attn_mask = (attention_mask == 0)

        all_outputs = [hidden] if return_all_steps else None

        # Recurrent refinement with big FFN
        for step in range(self.num_steps):
            delta = self.delta_refiner(hidden, attn_mask)
            hidden = hidden + delta

            if return_all_steps:
                all_outputs.append(hidden)

        if return_all_steps:
            return all_outputs
        else:
            return hidden

    def get_mlm_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> tuple:
        """Compute MLM loss for given hidden states."""
        logits = self.mlm_head(hidden)
        logits = logits.view(-1, self.vocab_size)
        labels = labels.view(-1)
        loss = F.cross_entropy(logits, labels, ignore_index=-100)
        return loss, logits.view(-1, logits.size(-1))

    def compute_recurrent_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        step_weights: list = None,
        return_accuracies: bool = False
    ) -> tuple:
        """Compute weighted loss across all refinement steps."""
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
        step_accs = [] if return_accuracies else None

        # Skip embedding (step 0), compute loss for refinement steps (1-4)
        for step_idx, hidden in enumerate(all_outputs[1:], start=0):
            loss, logits = self.get_mlm_loss(hidden, labels)
            weight = step_weights[step_idx]
            total_loss += weight * loss
            step_losses.append(loss.item())

            if return_accuracies:
                # Calculate accuracy for this step (detach to save memory)
                with torch.no_grad():
                    preds = logits.detach().argmax(dim=-1).view(-1)  # [B*L]
                    labels_flat = labels.view(-1)  # [B*L]
                    mask = (labels_flat != -100)  # [B*L]
                    correct = ((preds == labels_flat) & mask).sum().item()
                    total_tokens = mask.sum().item()
                    acc = correct / total_tokens if total_tokens > 0 else 0.0
                    step_accs.append(acc)

            # Delete logits immediately to free memory (~7.5GB per step)
            del logits

        if return_accuracies:
            return total_loss, step_losses, step_accs
        else:
            return total_loss, step_losses


def create_pnn_model(config: dict = None, model_type: str = 'pnn') -> nn.Module:
    """
    Factory function to create PNN model with config.

    Args:
        config: Model configuration dict
        model_type: Type of model ('pnn', 'pnn_exp1', 'pnn_exp2', 'pnn_exp3', 'pnn_exp4', 'pnn_exp5')

    Returns:
        model: PNN model instance
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

    # Create model based on type
    if model_type == 'pnn':
        model = PlasticNeuralNetwork(**config)
    elif model_type == 'pnn_exp1':
        # Exp1 uses dual attention + dual FFN structure
        model = PlasticNeuralNetworkExp1(**config)
    elif model_type == 'pnn_exp2':
        # Exp2 uses num_iterations instead of num_steps
        exp2_config = config.copy()
        exp2_config['num_iterations'] = exp2_config.pop('num_steps', 4) // 2
        model = PlasticNeuralNetworkExp2(**exp2_config)
    elif model_type == 'pnn_exp3':
        # Exp3 uses big single FFN (intermediate_size=4096)
        exp3_config = config.copy()
        exp3_config['intermediate_size'] = 4096
        model = PlasticNeuralNetworkExp3(**exp3_config)
    elif model_type == 'pnn_exp4':
        # Exp4 uses extended depth (3 transformer blocks)
        model = PlasticNeuralNetworkExp4(**config)
    elif model_type == 'pnn_exp5':
        # Exp5 uses 5 transformer blocks (BERT-baseline matched)
        model = PlasticNeuralNetworkExp5(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'pnn', 'pnn_exp1', 'pnn_exp2', 'pnn_exp3', 'pnn_exp4', 'pnn_exp5'")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Created {model_type.upper()} with {total_params:,} parameters ({total_params/1e6:.1f}M)")

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
