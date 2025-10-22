"""
first version hai TRM-Based Adaptive Embedding Scalable Retrieval System (TAESR) ka
Demo version with proper ML implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import time
import json
import pickle
from pathlib import Path
from collections import OrderedDict, deque
import logging
import re
import hashlib
from datetime import datetime
import unittest
from unittest.mock import Mock, patch


# Logging Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Config section

class ArchitectureType(Enum):
    MLP_MIXER = "mlp_mixer"  # Simple mixer
    SELF_ATTENTION = "self_attention"  # Attention wala
    HYBRID = "hybrid"  # Dono mix


@dataclass
class TRMConfig:
    """Complete TRM configuration"""
    # Architecture
    architecture: ArchitectureType = ArchitectureType.SELF_ATTENTION
    layers: int = 2
    hidden_dim: int = 512
    max_seq_length: int = 512
    vocab_size: int = 50000
    
    # Recursion
    max_recursion_steps: int = 6
    use_early_exit: bool = True
    early_exit_threshold: float = 0.98
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    ema_decay: float = 0.999
    batch_size: int = 32
    max_epochs: int = 100
    warmup_steps: int = 2000
    
    # Optimization
    use_gradient_checkpointing: bool = True
    mixed_precision: bool = True
    
    # Production
    cache_size: int = 10000
    enable_monitoring: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_recursion_steps <= 0:
            raise ValueError("max_recursion_steps must be positive")
        if not (0 <= self.early_exit_threshold <= 1):
            raise ValueError("early_exit_threshold must be between 0 and 1")
    
    def save(self, path: Union[str, Path]):
        """Save configuration to JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            config_dict = {
                'architecture': self.architecture.value,
                **{k: v for k, v in self.__dict__.items() if k != 'architecture'}
            }
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TRMConfig':
        """Load configuration from JSON"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
                config_dict['architecture'] = ArchitectureType(config_dict['architecture'])
                return cls(**config_dict)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Invalid config file: {e}")


# Tokenizer

class TRMTokenizer:
    """Tokenizer with vocab management"""
    
    def __init__(self, vocab_size: int = 50000, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
        self._initialize_vocab()
    
    def _initialize_vocab(self):
        """Initialize vocabulary with special tokens"""
        self.vocab = self.special_tokens.copy()
        self.inverse_vocab = {v: k for k, v in self.special_tokens.items()}
    
    def fit(self, texts: List[str]):
        """Build vocabulary from texts"""
        logger.info("Building vocabulary from texts...")
        word_freq = {}
        
        for text in texts:
            words = self._tokenize_text(text)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and take top vocab_size
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Add words to vocabulary
        for i, (word, _) in enumerate(sorted_words[:self.vocab_size - len(self.special_tokens)]):
            token_id = i + len(self.special_tokens)
            self.vocab[word] = token_id
            self.inverse_vocab[token_id] = word
        
        logger.info(f"Vocabulary built with {len(self.vocab)} tokens")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words with proper preprocessing"""
        # Convert to lowercase and split
        text = text.lower().strip()
        
        # Simple word tokenization with regex
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        
        return words
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        words = self._tokenize_text(text)
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.vocab['[CLS]'])
        
        for word in words:
            token_id = self.vocab.get(word, self.vocab['[UNK]'])
            tokens.append(token_id)
        
        if add_special_tokens:
            tokens.append(self.vocab['[SEP]'])
        
        # Truncate if necessary
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            if add_special_tokens:
                tokens[-1] = self.vocab['[SEP]']  # Ensure SEP at end
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        words = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                word = self.inverse_vocab[token_id]
                if word not in self.special_tokens:
                    words.append(word)
        
        return ' '.join(words)
    
    def save(self, path: Union[str, Path]):
        """Save tokenizer to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        tokenizer_data = {
            'vocab': self.vocab,
            'inverse_vocab': self.inverse_vocab,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'special_tokens': self.special_tokens
        }
        
        with open(path, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        
        logger.info(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TRMTokenizer':
        """Load tokenizer from file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        
        try:
            with open(path, 'rb') as f:
                tokenizer_data = pickle.load(f)
            
            tokenizer = cls(
                vocab_size=tokenizer_data['vocab_size'],
                max_length=tokenizer_data['max_length']
            )
            tokenizer.vocab = tokenizer_data['vocab']
            tokenizer.inverse_vocab = tokenizer_data['inverse_vocab']
            tokenizer.special_tokens = tokenizer_data['special_tokens']
            
            logger.info(f"Tokenizer loaded from {path}")
            return tokenizer
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer: {e}")


# NN components

class MLPMixerBlock(nn.Module):
    """MLP block, fixed length sequence ke liye"""
    
    def __init__(self, hidden_dim: int, seq_length: int):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(seq_length, seq_length),
            nn.GELU(),
            nn.Linear(seq_length, seq_length)
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input validation
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {x.dim()}D")
        
        # x: [batch, seq_len, hidden_dim]
        x = x + self.token_mixing(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mixing(x)
        return x


class TransformerBlock(nn.Module):
    """Self-attention transformer block"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input validation
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {x.dim()}D")
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        return x


class TinyRecursiveNetwork(nn.Module):
    """Core TRM recursive network - single tiny network for all recursion"""
    
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Input embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_dim)
        
        # Core recursive blocks (2 layers as per TRM paper)
        if config.architecture == ArchitectureType.MLP_MIXER:
            self.blocks = nn.ModuleList([
                MLPMixerBlock(config.hidden_dim, config.max_seq_length)
                for _ in range(config.layers)
            ])
        elif config.architecture == ArchitectureType.SELF_ATTENTION:
            self.blocks = nn.ModuleList([
                TransformerBlock(config.hidden_dim)
                for _ in range(config.layers)
            ])
        else:  # HYBRID
            self.blocks = nn.ModuleList([
                TransformerBlock(config.hidden_dim) if i == 0 else MLPMixerBlock(config.hidden_dim, config.max_seq_length)
                for i in range(config.layers)
            ])
        
        # Output heads
        self.output_head = nn.Linear(config.hidden_dim, config.vocab_size)
        self.halting_head = nn.Linear(config.hidden_dim, 1)  # For early exit
        
        # Layer norm
        self.norm = nn.LayerNorm(config.hidden_dim)
        
    def embed_input(self, x: torch.Tensor) -> torch.Tensor:
        """Create initial embeddings"""
        # Input validation
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D")
        
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        
        return token_emb + pos_emb
    
    def recursive_step(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single recursive refinement step"""
        # Input validation
        for tensor, name in [(x, 'x'), (y, 'y'), (z, 'z')]:
            if tensor.dim() != 3:
                raise ValueError(f"Expected 3D input for {name}, got {tensor.dim()}D")
            if tensor.shape != x.shape:
                raise ValueError(f"All inputs must have same shape, got {x.shape}, {y.shape}, {z.shape}")
        
        # Combine question (x), current answer (y), and latent reasoning (z)
        combined = x + y + z
        
        # Process through tiny network
        h = combined
        for block in self.blocks:
            if isinstance(block, MLPMixerBlock):
                h = block(h)
            else:  # TransformerBlock
                h = block(h)
        
        # Update latent reasoning z
        new_z = self.norm(h)
        
        # Update answer y (without x, as per TRM paper)
        answer_input = y + z
        h_answer = answer_input
        for block in self.blocks:
            if isinstance(block, MLPMixerBlock):
                h_answer = block(h_answer)
            else:
                h_answer = block(h_answer)
        new_y = self.norm(h_answer)
        
        return new_y, new_z
    
    def forward(self, x: torch.Tensor, max_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Full TRM forward pass with deep supervision"""
        # Input validation
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D")
        if x.size(1) > self.config.max_seq_length:
            raise ValueError(f"Sequence length {x.size(1)} exceeds max_seq_length {self.config.max_seq_length}")
        
        batch_size, seq_len = x.shape
        max_steps = max_steps or self.config.max_recursion_steps
        
        # Initial embeddings
        x_emb = self.embed_input(x)
        y = torch.zeros_like(x_emb)  # Initial answer
        z = torch.zeros_like(x_emb)  # Initial latent reasoning
        
        steps_taken = torch.zeros(batch_size, device=x.device, dtype=torch.long)
        converged = torch.zeros(batch_size, device=x.device, dtype=torch.bool)
        
        # Recursive refinement
        for step in range(max_steps):
            y_prev = y.clone()
            
            # Recursive step
            y, z = self.recursive_step(x_emb, y, z)
            
            # Early exit check
            if self.config.use_early_exit and step > 0:
                # Calculate similarity between consecutive y's
                similarity = F.cosine_similarity(
                    y.view(batch_size, -1), 
                    y_prev.view(batch_size, -1),
                    dim=1
                )
                
                newly_converged = (similarity > self.config.early_exit_threshold) & ~converged
                converged = converged | newly_converged
                steps_taken[newly_converged] = step + 1
                
                if converged.all():
                    break
        
        # Set steps for non-converged samples
        steps_taken[~converged] = max_steps
        
        # Generate final output
        output_logits = self.output_head(y)
        halting_probs = torch.sigmoid(self.halting_head(y))
        
        return {
            'logits': output_logits,
            'embeddings': y,
            'latent_reasoning': z,
            'halting_probs': halting_probs,
            'steps_taken': steps_taken,
            'converged': converged
        }


# Training system

class TRMTrainer:
    """Production training sys for trm"""
    
    def __init__(self, model: TinyRecursiveNetwork, config: TRMConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer (from TRM paper: AdamW)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # EMA (from TRM paper)
        self.ema_model = self._create_ema_model()
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.max_epochs
        )
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Model versioning
        self.current_version = 1
        self.version_history = []
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _create_ema_model(self) -> TinyRecursiveNetwork:
        """Create EMA model"""
        ema_model = TinyRecursiveNetwork(self.config)
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.eval()
        return ema_model
    
    def _update_ema(self):
        """Update EMA model weights"""
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.config.ema_decay).add_(
                    model_param.data, alpha=1 - self.config.ema_decay
                )
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        if not isinstance(train_loader, DataLoader):
            raise ValueError("train_loader must be a DataLoader instance")
        
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Validate batch
            if 'input_ids' not in batch or 'target_ids' not in batch:
                raise ValueError("Batch must contain 'input_ids' and 'target_ids'")
            
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                outputs = self.model(input_ids)
                
                # Cross-entropy loss
                loss = F.cross_entropy(
                    outputs['logits'].view(-1, self.config.vocab_size),
                    target_ids.view(-1),
                    ignore_index=-100
                )
                
                # Halting loss (for early exit training)
                if self.config.use_early_exit:
                    target_halt = (outputs['logits'].argmax(dim=-1) == target_ids).float()
                    halt_loss = F.binary_cross_entropy(
                        outputs['halting_probs'].squeeze(-1).mean(dim=1),
                        target_halt.any(dim=1).float()
                    )
                    loss = loss + 0.1 * halt_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update EMA
            self._update_ema()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                avg_steps = outputs['steps_taken'].float().mean().item()
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{num_batches}] "
                    f"Loss: {loss.item():.4f} Avg Steps: {avg_steps:.1f}"
                )
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        if not isinstance(val_loader, DataLoader):
            raise ValueError("val_loader must be a DataLoader instance")
        
        self.ema_model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        total_steps = []
        
        for batch in val_loader:
            # Validate batch
            if 'input_ids' not in batch or 'target_ids' not in batch:
                raise ValueError("Batch must contain 'input_ids' and 'target_ids'")
            
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            outputs = self.ema_model(input_ids)
            
            loss = F.cross_entropy(
                outputs['logits'].view(-1, self.config.vocab_size),
                target_ids.view(-1),
                ignore_index=-100
            )
            
            predictions = outputs['logits'].argmax(dim=-1)
            mask = target_ids != -100
            correct = (predictions == target_ids) & mask
            
            total_loss += loss.item()
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            total_steps.append(outputs['steps_taken'].float().mean().item())
        
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'accuracy': total_correct / total_tokens if total_tokens > 0 else 0,
            'avg_steps': np.mean(total_steps)
        }
        
        self.val_losses.append(metrics['val_loss'])
        return metrics
    
    def save_checkpoint(self, path: Union[str, Path], epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint with versioning"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update version info
        version_info = {
            'version': self.current_version,
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'metrics': metrics,
            'train_loss': self.train_losses[-1] if self.train_losses else 0,
            'val_loss': metrics.get('val_loss', 0)
        }
        self.version_history.append(version_info)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'version_info': version_info,
            'version_history': self.version_history
        }
        
        try:
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved to {path} (v{self.current_version})")
            
            # Save version history separately
            history_path = path.parent / 'version_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.version_history, f, indent=2)
            
            self.current_version += 1
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, path: Union[str, Path]) -> int:
        """Load training checkpoint"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            
            # Load version info if available
            if 'version_info' in checkpoint:
                self.current_version = checkpoint['version_info']['version'] + 1
            if 'version_history' in checkpoint:
                self.version_history = checkpoint['version_history']
            
            logger.info(f"Checkpoint loaded from {path}")
            return checkpoint['epoch']
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise


# Multi-Scale Embedding Cache

class PersistentEmbeddingCache:
    """with error handling"""
    
    def __init__(self, cache_dir: Union[str, Path], max_size: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        
        # Validate max_size
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        
        # In-memory caches
        self.shallow_cache = OrderedDict()  # LRU cache
        self.medium_cache = OrderedDict()
        self.deep_cache = OrderedDict()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.errors = 0
        
        # Load existing cache with error handling
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk with error handling"""
        for depth in ['shallow', 'medium', 'deep']:
            cache_file = self.cache_dir / f'{depth}_cache.pkl'
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                        # Convert list of tuples back to OrderedDict
                        cache_dict = OrderedDict(cache_data)
                        setattr(self, f'{depth}_cache', cache_dict)
                    logger.info(f"Loaded {len(cache_data)} entries from {depth} cache")
                except (pickle.PickleError, EOFError, KeyError) as e:
                    logger.error(f"Error loading {depth} cache: {e}")
                    self.errors += 1
                    # Initialize empty cache on error
                    setattr(self, f'{depth}_cache', OrderedDict())
            else:
                setattr(self, f'{depth}_cache', OrderedDict())
    
    def save_cache(self):
        """Save cache to disk with error handling"""
        for depth in ['shallow', 'medium', 'deep']:
            cache_file = self.cache_dir / f'{depth}_cache.pkl'
            cache = getattr(self, f'{depth}_cache')
            
            try:
                # Convert OrderedDict to list of tuples for pickle
                cache_data = list(cache.items())
                
                # Create temporary file for atomic write
                temp_file = cache_file.with_suffix('.tmp')
                with open(temp_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                
                # Atomic replace
                temp_file.replace(cache_file)
                logger.info(f"Saved {len(cache_data)} entries to {depth} cache")
                
            except (IOError, pickle.PickleError) as e:
                logger.error(f"Error saving {depth} cache: {e}")
                self.errors += 1
    
    def get(self, key: str, depth: str) -> Optional[torch.Tensor]:
        """Retrieve embedding from cache with validation"""
        if depth not in ['shallow', 'medium', 'deep']:
            raise ValueError("Depth must be 'shallow', 'medium', or 'deep'")
        
        if not isinstance(key, str):
            raise ValueError("Key must be a string")
        
        cache = getattr(self, f'{depth}_cache')
        
        if key in cache:
            # Move to end (most recently used)
            cache.move_to_end(key)
            self.hits += 1
            return cache[key]
        
        self.misses += 1
        return None
    
    def put(self, key: str, embedding: torch.Tensor, depth: str):
        """Store embedding in cache with validation"""
        if depth not in ['shallow', 'medium', 'deep']:
            raise ValueError("Depth must be 'shallow', 'medium', or 'deep'")
        
        if not isinstance(key, str):
            raise ValueError("Key must be a string")
        
        if not isinstance(embedding, torch.Tensor):
            raise ValueError("Embedding must be a torch.Tensor")
        
        cache = getattr(self, f'{depth}_cache')
        
        # Evict if full
        if len(cache) >= self.max_size:
            try:
                cache.popitem(last=False)  # Remove oldest
            except KeyError:
                pass  # Cache might be empty
        
        cache[key] = embedding.cpu().detach()  # Ensure tensor is on CPU and detached
    
    def clear_cache(self, depth: Optional[str] = None):
        """Clear cache with optional depth specification"""
        if depth is None:
            depths = ['shallow', 'medium', 'deep']
        else:
            if depth not in ['shallow', 'medium', 'deep']:
                raise ValueError("Depth must be 'shallow', 'medium', or 'deep'")
            depths = [depth]
        
        for depth in depths:
            cache = getattr(self, f'{depth}_cache')
            cache.clear()
        
        logger.info(f"Cleared cache for depths: {depths}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'errors': self.errors,
            'hit_rate': hit_rate,
            'shallow_size': len(self.shallow_cache),
            'medium_size': len(self.medium_cache),
            'deep_size': len(self.deep_cache),
            'total_size': len(self.shallow_cache) + len(self.medium_cache) + len(self.deep_cache)
        }


# Production Embedder

class ProductionTRMEmbedder:
    """TRM embedder with caching and adaptive depth"""
    
    def __init__(self, model: TinyRecursiveNetwork, config: TRMConfig, 
                 cache_dir: str, tokenizer: Optional[TRMTokenizer] = None):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Tokenizer
        self.tokenizer = tokenizer or self._create_default_tokenizer()
        
        # Cache
        self.cache = PersistentEmbeddingCache(cache_dir, config.cache_size)
        
        logger.info(f"Embedder initialized on {self.device}")
    
    def _create_default_tokenizer(self) -> TRMTokenizer:
        """Create a default tokenizer"""
        return TRMTokenizer(vocab_size=self.config.vocab_size, 
                           max_length=self.config.max_seq_length)
    
    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text using proper tokenizer"""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        
        tokens = self.tokenizer.encode(text)
        
        # Pad to max length
        if len(tokens) < self.config.max_seq_length:
            tokens.extend([self.tokenizer.vocab['[PAD]']] * 
                         (self.config.max_seq_length - len(tokens)))
        
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    def estimate_complexity(self, text: str) -> float:
        """Estimate text complexity for adaptive depth"""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        words = text.split()
        if not words:
            return 0.0
        
        # Length score
        length_score = min(len(words) / 50, 1.0)
        
        # Unique word ratio
        unique_ratio = len(set(words)) / len(words)
        
        # Question markers
        question_score = min(text.count('?') / 3, 1.0)
        
        # Sentence complexity (average word length)
        avg_word_length = np.mean([len(word) for word in words])
        complexity_score = min(avg_word_length / 10, 1.0)
        
        complexity = (0.3 * length_score + 0.25 * unique_ratio + 
                     0.25 * question_score + 0.2 * complexity_score)
        return min(complexity, 1.0)
    
    def get_optimal_steps(self, complexity: float) -> Tuple[int, str]:
        """Determine optimal recursion steps and cache depth"""
        if not (0 <= complexity <= 1):
            raise ValueError("Complexity must be between 0 and 1")
        
        if complexity < 0.3:
            steps = 2
            depth = 'shallow'
        elif complexity < 0.7:
            steps = min(4, self.config.max_recursion_steps)
            depth = 'medium'
        else:
            steps = self.config.max_recursion_steps
            depth = 'deep'
        
        return steps, depth
    
    @torch.no_grad()
    def embed(self, text: str, use_cache: bool = True, 
              adaptive: bool = True) -> Dict[str, Any]:
        """Generate embedding with caching and adaptive depth"""
        # Input validation
        if not isinstance(text, str):
            raise ValueError("Text must be a string")
        if not isinstance(use_cache, bool):
            raise ValueError("use_cache must be a boolean")
        if not isinstance(adaptive, bool):
            raise ValueError("adaptive must be a boolean")
        
        start_time = time.time()
        
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if use_cache:
            complexity = self.estimate_complexity(text)
            _, depth = self.get_optimal_steps(complexity)
            cached = self.cache.get(cache_key, depth)
            
            if cached is not None:
                return {
                    'embedding': cached.to(self.device),
                    'steps_used': 0,
                    'from_cache': True,
                    'latency_ms': (time.time() - start_time) * 1000,
                    'complexity': complexity,
                    'depth_category': depth,
                    'cache_key': cache_key
                }
        
        # Tokenize
        try:
            input_ids = self.tokenize(text).to(self.device)
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise
        
        # Determine steps
        if adaptive:
            complexity = self.estimate_complexity(text)
            steps, depth = self.get_optimal_steps(complexity)
        else:
            steps = self.config.max_recursion_steps
            depth = 'deep'
            complexity = 1.0
        
        # Generate embedding
        try:
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                outputs = self.model(input_ids, max_steps=steps)
            
            embedding = outputs['embeddings'].mean(dim=1)  # Pool over sequence
            
            # Cache result
            if use_cache:
                self.cache.put(cache_key, embedding, depth)
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                'embedding': embedding,
                'steps_used': outputs['steps_taken'][0].item(),
                'converged': outputs['converged'][0].item(),
                'from_cache': False,
                'latency_ms': latency_ms,
                'complexity': complexity,
                'depth_category': depth,
                'cache_key': cache_key
            }
        
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def batch_embed(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Batch embedding for efficiency"""
        if not isinstance(texts, list):
            raise ValueError("texts must be a list")
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All texts must be strings")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = [self.embed(text) for text in batch_texts]
            results.extend(batch_results)
        
        return results


# Data Handling

class TRMDataset(Dataset):
    """Dataset for TRM training"""

    def __init__(self, texts: List[str], tokenizer: TRMTokenizer, max_length: int = 512):
        if not isinstance(texts, list):
            raise ValueError("texts must be a list")
        if not isinstance(tokenizer, TRMTokenizer):
            raise ValueError("tokenizer must be a TRMTokenizer instance")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        
        self.texts = texts
        self.max_length = max_length
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self.texts):
            raise IndexError(f"Index {idx} out of range")
        
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)[:self.max_length]
        
        # Pad sequence
        if len(tokens) < self.max_length:
            tokens.extend([self.tokenizer.vocab['[PAD]']] * 
                         (self.max_length - len(tokens)))
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'target_ids': torch.tensor(tokens, dtype=torch.long)  # For self-supervised training
        }


# Live Update Manager

class LiveUpdateManager:
    """Manages live model updates and drift detection with error handling"""
    
    def __init__(self, model_dir: Union[str, Path], model: nn.Module, 
                 alert_threshold: float = 0.15):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.alert_threshold = alert_threshold
        self.performance_history = deque(maxlen=1000)
        self.update_history = []
        
        # Validate inputs
        if not isinstance(model, nn.Module):
            raise ValueError("model must be a nn.Module instance")
        if not (0 <= alert_threshold <= 1):
            raise ValueError("alert_threshold must be between 0 and 1")
        
    def track_performance(self, query_result: Dict[str, Any]):
        """Track query performance metrics with validation"""
        required_keys = ['latency_ms', 'steps_used', 'complexity']
        if not all(key in query_result for key in required_keys):
            raise ValueError(f"Query result must contain {required_keys}")
        
        self.performance_history.append({
            'timestamp': time.time(),
            'latency': query_result['latency_ms'],
            'steps': query_result['steps_used'],
            'complexity': query_result['complexity']
        })
        
    def detect_drift(self, window_size: int = 100) -> Optional[Dict[str, float]]:
        """Detect performance drift with validation"""
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        
        if len(self.performance_history) < window_size * 2:
            return None
            
        recent = list(self.performance_history)[-window_size:]
        baseline = list(self.performance_history)[-window_size*2:-window_size]
        
        if not baseline or not recent:
            return None
            
        try:
            # Calculate drift metrics
            latency_drift = (
                np.mean([r['latency'] for r in recent]) / 
                np.mean([b['latency'] for b in baseline]) - 1
            )
            
            steps_drift = (
                np.mean([r['steps'] for r in recent]) / 
                np.mean([b['steps'] for b in baseline]) - 1
            )
            
            drift_metrics = {
                'latency_drift': latency_drift,
                'steps_drift': steps_drift,
                'timestamp': time.time()
            }
            
            return drift_metrics if max(abs(latency_drift), abs(steps_drift)) > self.alert_threshold else None
        
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return None
    
    def trigger_update(self, drift_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Trigger model update based on drift with error handling"""
        if not isinstance(drift_metrics, dict):
            raise ValueError("drift_metrics must be a dictionary")
        
        update_info = {
            'timestamp': time.time(),
            'drift_metrics': drift_metrics,
            'status': 'initiated',
            'version': len(self.update_history) + 1
        }
        
        try:
            # Save current model as backup
            backup_path = self.model_dir / f'backup_v{update_info["version"]}_{int(time.time())}.pt'
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'drift_metrics': drift_metrics,
                'timestamp': update_info['timestamp']
            }, backup_path)
            
            # Log update
            update_info['status'] = 'completed'
            update_info['backup_path'] = str(backup_path)
            
            logger.info(f"Model update triggered (v{update_info['version']})")
            
        except Exception as e:
            update_info['status'] = 'failed'
            update_info['error'] = str(e)
            logger.error(f"Update failed: {e}")
            
        self.update_history.append(update_info)
        
        # Save update history
        self._save_update_history()
        
        return update_info
    
    def _save_update_history(self):
        """Save update history to file"""
        history_path = self.model_dir / 'update_history.json'
        try:
            with open(history_path, 'w') as f:
                json.dump(self.update_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save update history: {e}")
    
    def get_update_history(self) -> List[Dict[str, Any]]:
        """Get update history"""
        return self.update_history.copy()


# RAG system

class ProductionTRMRAG:
    """Fully loaded RAG"""
    
    def __init__(self, model_dir: Union[str, Path], cache_dir: Union[str, Path],
                 tokenizer: Optional[TRMTokenizer] = None):
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create directories if they don't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        config_path = self.model_dir / 'config.json'
        if not config_path.exists():
            # Create a default config if none exists
            logger.warning(f"Config not found at {config_path}, creating default config")
            default_config = TRMConfig()
            default_config.save(config_path)
            self.config = default_config
        else:
            self.config = TRMConfig.load(config_path)
         
        # Initialize model
        self.model = TinyRecursiveNetwork(self.config)
        self._load_model()
        
        # Initialize tokenizer
        self.tokenizer = tokenizer or TRMTokenizer(
            vocab_size=self.config.vocab_size,
            max_length=self.config.max_seq_length
        )
        
        # Initialize embedder
        self.embedder = ProductionTRMEmbedder(
            self.model, self.config, cache_dir, self.tokenizer
        )
        
        # Initialize update manager
        self.update_manager = LiveUpdateManager(
            model_dir=self.model_dir,
            model=self.model
        )
        
        # Document store
        self.documents = {}
        self.doc_embeddings = {}
        
        # Performance tracking
        self.query_stats = {
            'total_queries': 0,
            'total_latency_ms': 0,
            'cache_hits': 0,
            'avg_steps': []
        }
        
        logger.info("Production RAG system initialized")
    
    def _load_model(self):
        """Load trained model weights with error handling"""
        checkpoint_path = self.model_dir / 'best_model.pt'
        
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # Handle different checkpoint formats
                if 'ema_model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['ema_model_state_dict'])
                elif 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Assume it's directly the model state dict
                    self.model.load_state_dict(checkpoint)
                    
                logger.info(f"Model loaded from {checkpoint_path}")
                
                # Load version info if available
                if 'version_info' in checkpoint:
                    logger.info(f"Model version: {checkpoint['version_info']['version']}")
                    
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.warning("Using randomly initialized model")
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}, using random weights")
    
    def add_documents(self, documents: List[Dict[str, str]], batch_size: int = 32):
        """Add documents to the system with embeddings"""
        if not isinstance(documents, list):
            raise ValueError("documents must be a list")
        if not all(isinstance(doc, dict) and 'id' in doc and 'text' in doc for doc in documents):
            raise ValueError("Each document must be a dict with 'id' and 'text' keys")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        logger.info(f"Adding {len(documents)} documents...")
        
        # Check for duplicate IDs
        doc_ids = [doc['id'] for doc in documents]
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("Duplicate document IDs found")
        
        for doc in documents:
            doc_id = doc['id']
            self.documents[doc_id] = doc
        
        # Generate embeddings in batches
        texts = [doc['text'] for doc in documents]
        try:
            embeddings = self.embedder.batch_embed(texts, batch_size)
            
            for doc, emb_result in zip(documents, embeddings):
                self.doc_embeddings[doc['id']] = emb_result['embedding']
            
            logger.info(f"Documents indexed with embeddings")
            
        except Exception as e:
            logger.error(f"Failed to generate document embeddings: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5, use_adaptive: bool = True) -> Dict[str, Any]:
        """Search documents with TRM embeddings"""
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        start_time = time.time()
        
        # Generate query embedding
        query_result = self.embedder.embed(query, adaptive=use_adaptive)
        query_emb = query_result['embedding']
        
        # Calculate similarities
        similarities = {}
        for doc_id, doc_emb in self.doc_embeddings.items():
            sim = F.cosine_similarity(query_emb, doc_emb, dim=-1).item()
            similarities[doc_id] = sim
        
        # Rank documents
        ranked_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Prepare results
        results = []
        for doc_id, score in ranked_docs:
            results.append({
                'document': self.documents[doc_id],
                'score': float(score)
            })
        
        # Update stats
        total_time = (time.time() - start_time) * 1000
        self.query_stats['total_queries'] += 1
        self.query_stats['total_latency_ms'] += total_time
        if query_result['from_cache']:
            self.query_stats['cache_hits'] += 1
        self.query_stats['avg_steps'].append(query_result['steps_used'])
        
        result = {
            'results': results,
            'query_info': {
                'complexity': query_result['complexity'],
                'steps_used': query_result['steps_used'],
                'converged': query_result.get('converged', False),
                'from_cache': query_result['from_cache'],
                'latency_ms': query_result['latency_ms']
            },
            'total_latency_ms': total_time
        }
        
        # Track performance for drift detection
        self.update_manager.track_performance(query_result)
        
        # Check for drift
        drift = self.update_manager.detect_drift()
        if drift:
            logger.warning(f"Performance drift detected: {drift}")
            update_result = self.update_manager.trigger_update(drift)
            result['drift_detected'] = True
            result['update_info'] = update_result
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        stats = self.query_stats.copy()
        
        if stats['total_queries'] > 0:
            stats['avg_latency_ms'] = stats['total_latency_ms'] / stats['total_queries']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_queries']
            stats['avg_recursion_steps'] = np.mean(stats['avg_steps']) if stats['avg_steps'] else 0
        else:
            stats['avg_latency_ms'] = 0
            stats['cache_hit_rate'] = 0
            stats['avg_recursion_steps'] = 0
        
        # Add cache statistics
        stats['cache_stats'] = self.embedder.cache.get_stats()
        
        # Add model version info
        stats['update_history_count'] = len(self.update_manager.get_update_history())
        
        return stats
    
    def save_system_state(self, path: Union[str, Path]):
        """Save complete system state"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        system_state = {
            'documents': self.documents,
            'doc_embeddings': {k: v.cpu().numpy().tolist() for k, v in self.doc_embeddings.items()},
            'query_stats': self.query_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(system_state, f, indent=2)
            logger.info(f"System state saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save system state: {e}")
            raise


# Deployment Manager

class TRMDeploymentManager:
    """Manages TRM deployment, monitoring, and health checks"""
    
    def __init__(self, rag_system: ProductionTRMRAG):
        self.rag_system = rag_system
        self.start_time = time.time()
        
        # Validate input
        if not isinstance(rag_system, ProductionTRMRAG):
            raise ValueError("rag_system must be a ProductionTRMRAG instance")
        
        # Monitoring
        self.query_log = deque(maxlen=10000)
        self.error_log = deque(maxlen=1000)
        self.health_metrics = {
            'gpu_memory': deque(maxlen=100),
            'cache_stats': deque(maxlen=100),
            'query_latencies': deque(maxlen=1000)
        }
    
    def monitor_query(self, query: str, result: Dict[str, Any]):
        """Monitor individual query performance with validation"""
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if not isinstance(result, dict):
            raise ValueError("result must be a dictionary")
        
        query_info = {
            'timestamp': time.time(),
            'query': query[:100] + '...' if len(query) > 100 else query,  # Truncate long queries
            'latency': result['total_latency_ms'],
            'cache_hit': result['query_info']['from_cache'],
            'steps': result['query_info']['steps_used'],
            'results_count': len(result['results'])
        }
        self.query_log.append(query_info)
        self.health_metrics['query_latencies'].append(result['total_latency_ms'])
        
        # Track in update manager
        self.rag_system.update_manager.track_performance(result['query_info'])
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        uptime = time.time() - self.start_time
        recent_queries = list(self.query_log)[-100:] if self.query_log else []
        
        health_status = {
            'status': 'healthy',
            'uptime_hours': uptime / 3600,
            'query_stats': {
                'total_queries': len(self.query_log),
                'recent_queries': len(recent_queries),
            },
            'cache_stats': self.rag_system.embedder.cache.get_stats(),
            'drift_detected': self.rag_system.update_manager.detect_drift() is not None,
            'system_stats': self.rag_system.get_statistics()
        }
        
        # Add latency metrics if we have recent queries
        if recent_queries:
            latencies = [q['latency'] for q in recent_queries]
            health_status['query_stats'].update({
                'avg_latency': np.mean(latencies),
                'p95_latency': np.percentile(latencies, 95),
                'cache_hit_rate': np.mean([q['cache_hit'] for q in recent_queries])
            })
        
        # Check for critical issues
        if health_status['cache_stats']['errors'] > 10:
            health_status['status'] = 'degraded'
            health_status['issues'] = ['High cache error rate']
        
        if health_status['drift_detected']:
            health_status['status'] = 'needs_attention'
            health_status['issues'] = ['Performance drift detected']
        
        return health_status
    
    def export_monitoring_report(self, output_path: Union[str, Path]):
        """Export monitoring data as JSON report"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': time.time(),
            'health_status': self.get_health_status(),
            'recent_queries': list(self.query_log)[-100:],
            'recent_errors': list(self.error_log)[-100:],
            'update_history': self.rag_system.update_manager.get_update_history()
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Monitoring report exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export monitoring report: {e}")
            raise


# Testing time

class TestTRMSystem(unittest.TestCase):
    """Sab kuch test karo"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = TRMConfig(
            hidden_dim=64,
            max_seq_length=32,
            vocab_size=1000,
            max_recursion_steps=3
        )
        self.model = TinyRecursiveNetwork(self.config)
        self.tokenizer = TRMTokenizer(vocab_size=1000, max_length=32)
        
        # Sample data for tokenizer
        sample_texts = ["hello world", "test sentence", "another example"]
        self.tokenizer.fit(sample_texts)
    
    def test_tokenizer_encoding_decoding(self):
        """Test tokenizer encoding and decoding"""
        text = "hello world test"
        tokens = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(tokens)
        
        # Should recover original words (order may change due to tokenization)
        original_words = set(text.lower().split())
        decoded_words = set(decoded.split())
        self.assertTrue(original_words.issubset(decoded_words))
    
    def test_tokenizer_vocab_size(self):
        """Test tokenizer vocabulary size"""
        self.assertLessEqual(len(self.tokenizer.vocab), self.tokenizer.vocab_size)
    
    def test_model_forward_pass(self):
        """Test model forward pass with valid input"""
        input_tensor = torch.randint(0, 1000, (2, 16))
        output = self.model(input_tensor)
        
        self.assertIn('logits', output)
        self.assertIn('embeddings', output)
        self.assertIn('steps_taken', output)
        
        self.assertEqual(output['logits'].shape, (2, 16, self.config.vocab_size))
        self.assertEqual(output['embeddings'].shape, (2, 16, self.config.hidden_dim))
    
    def test_model_invalid_input(self):
        """Test model with invalid input"""
        with self.assertRaises(ValueError):
            # 3D input instead of 2D
            input_tensor = torch.randint(0, 1000, (2, 16, 8))
            self.model(input_tensor)
    
    def test_embedder_basic_functionality(self):
        """Test embedder basic functionality"""
        embedder = ProductionTRMEmbedder(
            self.model, self.config, "test_cache", self.tokenizer
        )
        
        result = embedder.embed("test sentence", use_cache=False)
        
        self.assertIn('embedding', result)
        self.assertIn('steps_used', result)
        self.assertIn('complexity', result)
        self.assertEqual(result['from_cache'], False)
    
    def test_cache_operations(self):
        """Test cache operations"""
        cache = PersistentEmbeddingCache("test_cache", max_size=10)
        
        # Test put and get
        test_tensor = torch.randn(1, 64)
        cache.put("test_key", test_tensor, "shallow")
        retrieved = cache.get("test_key", "shallow")
        
        self.assertIsNotNone(retrieved)
        self.assertTrue(torch.allclose(test_tensor, retrieved))
        
        # Test cache stats
        stats = cache.get_stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['shallow_size'], 1)
    
    def test_config_validation(self):
        """Test configuration validation"""
        with self.assertRaises(ValueError):
            TRMConfig(hidden_dim=-1)  # Invalid hidden_dim
        
        with self.assertRaises(ValueError):
            TRMConfig(early_exit_threshold=1.5)  # Invalid threshold
    
    def test_rag_system_search(self):
        """Test RAG system search functionality"""
        rag_system = ProductionTRMRAG(
            model_dir="test_model",
            cache_dir="test_cache",
            tokenizer=self.tokenizer
        )
        
        # Add test documents
        documents = [
            {'id': '1', 'text': 'hello world'},
            {'id': '2', 'text': 'test document'}
        ]
        rag_system.add_documents(documents)
        
        # Test search
        results = rag_system.search("hello", top_k=2)
        
        self.assertIn('results', results)
        self.assertIn('query_info', results)
        self.assertEqual(len(results['results']), 2)
    
    def tearDown(self):
        """Clean up test files"""
        import shutil
        
        # Remove test directories
        test_dirs = ["test_cache", "test_model"]
        for dir_path in test_dirs:
            if Path(dir_path).exists():
                shutil.rmtree(dir_path)


# Example Usage

def example_usage():
    """Example usage of the complete TRM system"""
    print("=== TRM System Example Usage ===")
    
    # Create necessary directories
    Path("example_model").mkdir(exist_ok=True)
    Path("example_cache").mkdir(exist_ok=True)
    Path("test_model").mkdir(exist_ok=True)
    Path("test_cache").mkdir(exist_ok=True)
    
    # Initialize model and config
    config = TRMConfig(
        hidden_dim=128,
        max_seq_length=64,
        vocab_size=5000,
        max_recursion_steps=4,
        use_gradient_checkpointing=False,  # Disable for CPU
        mixed_precision=False  # Disable for CPU
    )
    
    # Save config for RAG system
    config.save("example_model/config.json")
    
    # Create and fit tokenizer
    tokenizer = TRMTokenizer(vocab_size=5000, max_length=64)
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require large amounts of training data."
    ]
    tokenizer.fit(sample_texts)
    
    # Initialize model
    model = TinyRecursiveNetwork(config)
    
    # Training example (simplified)
    print("1. Model training setup completed")
    
    # Inference example
    embedder = ProductionTRMEmbedder(model, config, "example_cache", tokenizer)
    result = embedder.embed("Sample text for embedding generation")
    print(f"2. Embedding generated - Steps used: {result['steps_used']}, "
          f"Complexity: {result['complexity']:.3f}")
    
    # RAG system example
    rag_system = ProductionTRMRAG(
        model_dir="example_model",
        cache_dir="example_cache",
        tokenizer=tokenizer
    )
    
    # Add documents
    documents = [
        {"id": "doc1", "text": "Artificial intelligence is transforming many industries."},
        {"id": "doc2", "text": "Machine learning algorithms learn from data patterns."},
        {"id": "doc3", "text": "Deep learning uses neural networks with multiple layers."}
    ]
    rag_system.add_documents(documents)
    
    # Search
    search_results = rag_system.search("What is machine learning?")
    print(f"3. Search completed - Found {len(search_results['results'])} documents")
    
    # Monitoring example
    deployment = TRMDeploymentManager(rag_system)
    health = deployment.get_health_status()
    print(f"4. System health: {health['status']}")
    
    # Run unit tests
    print("5. Running unit tests...")
    
    # Create test config first
    test_config = TRMConfig(
        hidden_dim=64,
        max_seq_length=32,
        vocab_size=1000,
        max_recursion_steps=3,
        use_gradient_checkpointing=False,
        mixed_precision=False
    )
    test_config.save("test_model/config.json")
    
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestTRMSystem)
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    print("=== Example completed ===")


if __name__ == "__main__":
    example_usage()