"""
TAESR Training Script
Complete training pipeline with distributed training support, checkpointing, and evaluation.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Tuple, Optional
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
import wandb
from TAESRv2 import TAESRLoss

logger = logging.getLogger(__name__)


# TRAINING CONFIGURATION

@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # Model settings
    model_config: Dict = field(default_factory=dict)
    
    # Data settings
    train_data_path: str = "data/train.jsonl"
    eval_data_path: str = "data/eval.jsonl"
    max_seq_length: int = 512
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimization
    use_fp16: bool = True
    use_bf16: bool = False
    
    # Loss weights
    contrastive_temp: float = 0.05
    router_weight: float = 0.1
    splade_weight: float = 0.05
    multidim_weight: float = 0.1
    
    # Distributed training
    local_rank: int = -1
    world_size: int = 1
    
    # Checkpointing
    output_dir: str = "outputs/taesr"
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "taesr-training"
    wandb_run_name: Optional[str] = None
    
    # Evaluation
    eval_batch_size: int = 64
    eval_during_training: bool = True
    
    # Random seed
    seed: int = 42


# DATASET

class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning with query-document pairs.
    
    Format: Each line is a JSON object with:
    {
        "query": "text",
        "positive": "text",
        "negative": "text" (optional),
        "complexity": "easy/medium/hard" (optional)
    }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        include_negatives: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_negatives = include_negatives
        
        # Load data
        self.examples = []
        with open(data_path, 'r') as f:
            for line in f:
                self.examples.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize query and positive
        query_inputs = self.tokenizer(
            example['query'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        positive_inputs = self.tokenizer(
            example['positive'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        query_inputs = {k: v.squeeze(0) for k, v in query_inputs.items()}
        positive_inputs = {k: v.squeeze(0) for k, v in positive_inputs.items()}
        
        # Complexity label (for router supervision)
        complexity_map = {'easy': 0, 'medium': 1, 'hard': 2}
        complexity = complexity_map.get(example.get('complexity', 'medium'), 1)
        
        return {
            'query_input_ids': query_inputs['input_ids'],
            'query_attention_mask': query_inputs['attention_mask'],
            'positive_input_ids': positive_inputs['input_ids'],
            'positive_attention_mask': positive_inputs['attention_mask'],
            'complexity_label': torch.tensor(complexity, dtype=torch.long)
        }


def collate_fn(batch):
    """Custom collate function for batching."""
    return {
        'query_input_ids': torch.stack([x['query_input_ids'] for x in batch]),
        'query_attention_mask': torch.stack([x['query_attention_mask'] for x in batch]),
        'positive_input_ids': torch.stack([x['positive_input_ids'] for x in batch]),
        'positive_attention_mask': torch.stack([x['positive_attention_mask'] for x in batch]),
        'complexity_labels': torch.stack([x['complexity_label'] for x in batch])
    }


# TRAINER

class TAESRTrainer:
    """
    Complete training pipeline for TAESR model.
    Supports distributed training, mixed precision, and comprehensive logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        config: TrainingConfig,
        tokenizer=None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.tokenizer = tokenizer
        
        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Setup distributed training
        if config.local_rank != -1:
            self.model = DDP(
                self.model,
                device_ids=[config.local_rank],
                output_device=config.local_rank,
                find_unused_parameters=True
            )
        
        # Setup dataloaders
        self.train_loader = self._create_dataloader(train_dataset, config.batch_size, shuffle=True)
        self.eval_loader = self._create_dataloader(eval_dataset, config.eval_batch_size, shuffle=False) if eval_dataset else None
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = TAESRLoss(
            model.config if not isinstance(model, DDP) else model.module.config,
            temperature=config.contrastive_temp,
            router_weight=config.router_weight,
            splade_weight=config.splade_weight,
            multidim_weight=config.multidim_weight
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_fp16 else None
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Logging
        if config.use_wandb and (config.local_rank in [-1, 0]):
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.__dict__
            )
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ Trainer initialized on {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.local_rank != -1:
            torch.cuda.set_device(self.config.local_rank)
            device = torch.device("cuda", self.config.local_rank)
            dist.init_process_group(backend='nccl')
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device
    
    def _create_dataloader(self, dataset, batch_size, shuffle=True):
        """Create dataloader with optional distributed sampler."""
        if dataset is None:
            return None
        
        sampler = None
        if self.config.local_rank != -1:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=self.config.local_rank,
                shuffle=shuffle
            )
            shuffle = False
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
    
    def _create_optimizer(self):
        """Create AdamW optimizer with weight decay."""
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        num_training_steps = len(self.train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train(self):
        """Main training loop."""
        logger.info("🚀 Starting training...")
        logger.info(f"   Num examples: {len(self.train_dataset)}")
        logger.info(f"   Num epochs: {self.config.num_epochs}")
        logger.info(f"   Batch size: {self.config.batch_size}")
        logger.info(f"   Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"   Total steps: {len(self.train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps}")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            self._train_epoch()
            
            # Evaluate at end of epoch
            if self.eval_loader and self.config.eval_during_training:
                eval_results = self.evaluate()
                logger.info(f"Epoch {epoch} evaluation: {eval_results}")
            
            # Save checkpoint
            if self.config.local_rank in [-1, 0]:
                self._save_checkpoint(f"checkpoint-epoch-{epoch}")
        
        logger.info("✅ Training completed!")
    
    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch}",
            disable=self.config.local_rank not in [-1, 0]
        )
        
        for step, batch in enumerate(progress_bar):
            loss = self._training_step(batch)
            epoch_loss += loss
            
            # Update progress bar
            if step % self.config.logging_steps == 0:
                progress_bar.set_postfix({'loss': f'{loss:.4f}'})
            
            # Logging
            if self.global_step % self.config.logging_steps == 0 and self.config.local_rank in [-1, 0]:
                self._log_metrics({'train/loss': loss, 'train/lr': self.scheduler.get_last_lr()[0]})
            
            # Evaluation
            if self.config.eval_during_training and self.global_step % self.config.eval_steps == 0:
                eval_results = self.evaluate()
                if self.config.local_rank in [-1, 0]:
                    self._log_metrics({f'eval/{k}': v for k, v in eval_results.items()})
                self.model.train()
            
            # Checkpointing
            if self.global_step % self.config.save_steps == 0 and self.config.local_rank in [-1, 0]:
                self._save_checkpoint(f"checkpoint-step-{self.global_step}")
        
        avg_loss = epoch_loss / len(self.train_loader)
        logger.info(f"Epoch {self.epoch} average loss: {avg_loss:.4f}")
    
    def _training_step(self, batch) -> float:
        """Execute one training step."""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with autocast(enabled=self.config.use_fp16):
            # Query forward
            query_outputs = self.model(
                input_ids=batch['query_input_ids'],
                attention_mask=batch['query_attention_mask'],
                return_dict=True
            )
            
            # Positive forward
            positive_outputs = self.model(
                input_ids=batch['positive_input_ids'],
                attention_mask=batch['positive_attention_mask'],
                return_dict=True
            )
            
            # Compute loss
            losses = self.criterion(
                query_outputs,
                positive_outputs,
                complexity_labels=batch.get('complexity_labels')
            )
            
            loss = losses['total'] / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        self.global_step += 1
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        if self.eval_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.eval_loader, desc="Evaluating", disable=self.config.local_rank not in [-1, 0]):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            query_outputs = self.model(
                input_ids=batch['query_input_ids'],
                attention_mask=batch['query_attention_mask'],
                return_dict=True
            )
            
            positive_outputs = self.model(
                input_ids=batch['positive_input_ids'],
                attention_mask=batch['positive_attention_mask'],
                return_dict=True
            )
            
            # Compute loss
            losses = self.criterion(query_outputs, positive_outputs)
            total_loss += losses['total'].item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Update best model
        if avg_loss < self.best_eval_loss:
            self.best_eval_loss = avg_loss
            if self.config.local_rank in [-1, 0]:
                self._save_checkpoint("best_model")
        
        return {'loss': avg_loss}
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        output_path = Path(self.config.output_dir) / checkpoint_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get model (unwrap DDP if needed)
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        
        # Save model
        model_to_save.save_pretrained(output_path)
        
        # Save training state
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_eval_loss': self.best_eval_loss,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }, output_path / 'training_state.pt')
        
        # Save config
        with open(output_path / 'training_config.json', 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"💾 Checkpoint saved to {output_path}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        checkpoints = sorted(Path(self.config.output_dir).glob("checkpoint-step-*"))
        
        if len(checkpoints) > self.config.save_total_limit:
            for checkpoint in checkpoints[:-self.config.save_total_limit]:
                import shutil
                shutil.rmtree(checkpoint)
                logger.info(f"🗑️  Removed old checkpoint: {checkpoint}")
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to console and W&B."""
        if self.config.use_wandb:
            wandb.log(metrics, step=self.global_step)


# MAIN TRAINING SCRIPT

def main():
    """Main entry point for training."""
    # Parse arguments (simplified - use argparse in production)
    config = TrainingConfig(
        train_data_path="data/train.jsonl",
        eval_data_path="data/eval.jsonl",
        num_epochs=3,
        batch_size=32,
        learning_rate=2e-5,
        output_dir="outputs/taesr",
        use_wandb=False
    )
    
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Initialize model
    from TAESRv2 import TAESRModel, TAESRConfig
    
    model_config = TAESRConfig(
        hidden_size=384,
        num_attention_heads=6,
        intermediate_size=1536,
        recursion_steps={"easy": 1, "medium": 3, "hard": 6}
    )
    
    model = TAESRModel(model_config)
    logger.info(f"Model initialized: {model.num_parameters():,} parameters")
    
    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Load datasets (placeholder - implement actual data loading)
    train_dataset = ContrastiveDataset(config.train_data_path, tokenizer)
    eval_dataset = ContrastiveDataset(config.eval_data_path, tokenizer)
    
    # Initialize trainer
    trainer = TAESRTrainer(model, train_dataset, eval_dataset, config)
    
    # Start training
    trainer.train()
    
    logger.info("✅ Training script ready for execution")


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    main()