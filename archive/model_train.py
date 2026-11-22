"""
Training Script for Custom Empathy Analyzer Model
==================================================
Multi-task training with:
- Emotion classification (multi-label, 20 classes)
- Empathy score regression ([-3, +3])
- Communication style classification (8 classes)
- IMDb rating prediction (movie-level, [1, 10])

Supports phased training, gradient accumulation, mixed precision,
and comprehensive logging.
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from model.custom_model import (
    EmpathyAnalyzerModel, ModelConfig, create_model,
    EMOTION_LABELS, STYLE_LABELS
)
from model.tokenizer import ScreenplayBPETokenizer, train_tokenizer_from_screenplays
from model.dataset import (
    ScreenplayDataset, MovieLevelDataset,
    create_dataloaders, create_movie_dataloaders,
    collate_dialogue_batch, collate_movie_batch
)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Data
    data_path: str = "data/labeled/dialogues.jsonl"
    imdb_ratings_path: str = "data/imdb_ratings.json"
    tokenizer_path: str = "cache/tokenizer.pkl"
    screenplay_dir: str = "data for tranning"

    # Model
    vocab_size: int = 12000
    embed_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.3
    max_seq_len: int = 256

    # Training
    batch_size: int = 32
    movie_batch_size: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_epochs: int = 50
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    eval_every: int = 500
    save_every: int = 1000

    # Multi-task loss weights
    lambda_emotion: float = 1.0
    lambda_empathy: float = 1.0
    lambda_style: float = 1.0
    lambda_imdb: float = 0.5

    # Phased training
    phase1_epochs: int = 10      # Emotion + Empathy only
    phase2_epochs: int = 15      # + Style
    phase3_epochs: int = 25      # + IMDb (movie-level)

    # Technical
    device: str = "auto"         # auto, cuda, cpu
    mixed_precision: bool = True
    num_workers: int = 0
    seed: int = 42

    # Output
    output_dir: str = "checkpoints"
    experiment_name: str = "empathy_analyzer"

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            attention_heads=self.num_heads,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
            lambda_emotion=self.lambda_emotion,
            lambda_empathy=self.lambda_empathy,
            lambda_style=self.lambda_style,
            lambda_imdb=self.lambda_imdb,
        )


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def prepare_tokenizer(config: TrainingConfig) -> ScreenplayBPETokenizer:
    """Load or train tokenizer."""
    tokenizer_path = Path(config.tokenizer_path)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    if tokenizer_path.exists():
        print(f"Loading tokenizer from {tokenizer_path}")
        return ScreenplayBPETokenizer.load(str(tokenizer_path))

    print("Training new tokenizer...")
    tokenizer = train_tokenizer_from_screenplays(
        config.screenplay_dir,
        str(tokenizer_path),
        vocab_size=config.vocab_size
    )
    return tokenizer


def prepare_imdb_ratings(config: TrainingConfig) -> Dict[str, float]:
    """Load or create IMDb ratings file."""
    ratings_path = Path(config.imdb_ratings_path)
    ratings_path.parent.mkdir(parents=True, exist_ok=True)

    if ratings_path.exists():
        with open(ratings_path, 'r') as f:
            return json.load(f)

    # Default ratings for known movies in the dataset
    default_ratings = {
        "AMERICAN_PSYCHO": 7.6,
        "SW_EpisodeIV": 8.6,
        "SW_EpisodeV": 8.7,
        "SW_EpisodeVI": 8.3,
    }

    with open(ratings_path, 'w') as f:
        json.dump(default_ratings, f, indent=2)

    print(f"Created default IMDb ratings at {ratings_path}")
    return default_ratings


def train_dialogue_phase(
    model: EmpathyAnalyzerModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    device: torch.device,
    config: TrainingConfig,
    phase_name: str,
    num_epochs: int,
    global_step: int,
    best_val_loss: float,
    output_dir: Path
) -> Tuple[int, float]:
    """Train dialogue-level tasks (emotion, empathy, style)."""

    model.train()
    epoch_pbar = tqdm(range(num_epochs), desc=f"{phase_name} Epochs")

    for epoch in epoch_pbar:
        epoch_losses = {"total": 0.0, "emotion": 0.0, "empathy": 0.0, "style": 0.0}
        num_batches = 0

        batch_pbar = tqdm(train_loader, desc=f"{phase_name} Batches", leave=False)

        for batch in batch_pbar:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emotion_labels = batch["emotion_labels"].to(device)
            empathy_scores = batch["empathy_scores"].to(device)
            style_labels = batch["style_labels"].to(device)
            confidence = batch["confidence"].to(device)

            # Forward with mixed precision
            with autocast(enabled=config.mixed_precision):
                outputs = model(input_ids, attention_mask)

                targets = {
                    "emotion_labels": emotion_labels,
                    "empathy_scores": empathy_scores,
                    "style_labels": style_labels
                }

                loss, loss_dict = model.get_loss(outputs, targets)

                # Apply confidence weighting
                loss = loss * confidence.mean()

            # Backward
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (num_batches + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            # Accumulate losses
            for k, v in loss_dict.items():
                epoch_losses[k] += v
            num_batches += 1

            # Log
            batch_pbar.set_postfix({
                "loss": f"{loss_dict['total']:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })

            # Evaluation
            if global_step % config.eval_every == 0:
                val_loss = evaluate_dialogue(model, val_loader, device, config)
                model.train()
                print(f"\nStep {global_step} | Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, scheduler, global_step, best_val_loss,
                                  output_dir / f"{config.experiment_name}_best.pt")
                    print(f"  ✓ New best model saved!")

            # Checkpoint
            if global_step % config.save_every == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, best_val_loss,
                              output_dir / f"{config.experiment_name}_step{global_step}.pt")

        # Epoch summary
        avg_losses = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
        epoch_pbar.set_postfix(avg_losses)

    return global_step, best_val_loss


def evaluate_dialogue(
    model: EmpathyAnalyzerModel,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig
) -> float:
    """Evaluate dialogue-level tasks."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    all_emotion_preds = []
    all_emotion_targets = []
    all_empathy_preds = []
    all_empathy_targets = []
    all_style_preds = []
    all_style_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emotion_labels = batch["emotion_labels"].to(device)
            empathy_scores = batch["empathy_scores"].to(device)
            style_labels = batch["style_labels"].to(device)

            with autocast(enabled=config.mixed_precision):
                outputs = model(input_ids, attention_mask)

                targets = {
                    "emotion_labels": emotion_labels,
                    "empathy_scores": empathy_scores,
                    "style_labels": style_labels
                }

                loss, _ = model.get_loss(outputs, targets)

            total_loss += loss.item()
            total_batches += 1

            # Collect predictions for metrics
            all_emotion_preds.append(torch.sigmoid(outputs["emotion_logits"]).cpu())
            all_emotion_targets.append(emotion_labels.cpu())
            all_empathy_preds.append(outputs["empathy_scores"].cpu())
            all_empathy_targets.append(empathy_scores.cpu())
            all_style_preds.append(outputs["style_logits"].argmax(dim=-1).cpu())
            all_style_targets.append(style_labels.cpu())

    # Compute metrics
    if all_emotion_preds:
        emotion_preds = torch.cat(all_emotion_preds)
        emotion_targets = torch.cat(all_emotion_targets)
        empathy_preds = torch.cat(all_empathy_preds)
        empathy_targets = torch.cat(all_empathy_targets)
        style_preds = torch.cat(all_style_preds)
        style_targets = torch.cat(all_style_targets)

        # Emotion: F1 per class (threshold 0.5)
        emotion_binary = (emotion_preds > 0.5).float()
        emotion_f1 = compute_f1_per_class(emotion_binary, emotion_targets, EMOTION_LABELS)

        # Empathy: MSE, MAE, Correlation
        empathy_mse = torch.nn.functional.mse_loss(empathy_preds, empathy_targets).item()
        empathy_mae = torch.nn.functional.l1_loss(empathy_preds, empathy_targets).item()
        empathy_corr = torch.corrcoef(torch.stack([empathy_preds, empathy_targets]))[0, 1].item()

        # Style: Accuracy
        style_acc = (style_preds == style_targets).float().mean().item()

        print(f"\n  Emotion Macro F1: {emotion_f1['macro']:.4f}")
        print(f"  Empathy MSE: {empathy_mse:.4f}, MAE: {empathy_mae:.4f}, Corr: {empathy_corr:.4f}")
        print(f"  Style Acc: {style_acc:.4f}")

    return total_loss / max(total_batches, 1)


def compute_f1_per_class(preds: torch.Tensor, targets: torch.Tensor, labels: List[str]) -> Dict:
    """Compute F1 score per class for multi-label classification."""
    f1s = {}
    for i, label in enumerate(labels):
        tp = ((preds[:, i] == 1) & (targets[:, i] == 1)).sum().item()
        fp = ((preds[:, i] == 1) & (targets[:, i] == 0)).sum().item()
        fn = ((preds[:, i] == 0) & (targets[:, i] == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1s[label] = f1

    f1s["macro"] = sum(f1s.values()) / len(f1s)
    return f1s


def train_movie_phase(
    model: EmpathyAnalyzerModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    device: torch.device,
    config: TrainingConfig,
    global_step: int,
    best_val_loss: float,
    output_dir: Path
) -> Tuple[int, float]:
    """Train movie-level IMDb rating prediction."""

    model.train()
    num_epochs = config.phase3_epochs
    epoch_pbar = tqdm(range(num_epochs), desc="IMDb Phase Epochs")

    for epoch in epoch_pbar:
        epoch_losses = {"total": 0.0, "imdb": 0.0, "emotion": 0.0, "empathy": 0.0, "style": 0.0}
        num_batches = 0

        for batch in train_loader:
            # batch is a list of movie dicts (batch_size=1)
            movie_batch = batch[0]

            input_ids = movie_batch["input_ids"].to(device)           # (num_dialogues, max_len)
            attention_mask = movie_batch["attention_mask"].to(device)
            emotion_labels = movie_batch["emotion_labels"].to(device)
            empathy_scores = movie_batch["empathy_scores"].to(device)
            style_labels = movie_batch["style_labels"].to(device)
            character_ids = movie_batch["character_ids"].to(device)
            num_characters = movie_batch["num_characters"]
            imdb_target = movie_batch["imdb_rating"].to(device)

            with autocast(enabled=config.mixed_precision):
                outputs = model(
                    input_ids, attention_mask,
                    character_ids=character_ids,
                    num_characters=num_characters,
                    return_char_profiles=True
                )

                targets = {
                    "emotion_labels": emotion_labels,
                    "empathy_scores": empathy_scores,
                    "style_labels": style_labels
                }

                loss, loss_dict = model.get_loss(
                    outputs, targets,
                    character_profiles=outputs.get("character_profiles"),
                    imdb_target=imdb_target.unsqueeze(0)
                )

            scaler.scale(loss).backward()

            if (num_batches + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            for k, v in loss_dict.items():
                epoch_losses[k] += v
            num_batches += 1

        avg_losses = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
        epoch_pbar.set_postfix(avg_losses)

        # Validation
        val_loss = evaluate_movie(model, val_loader, device, config)
        model.train()
        print(f"\nEpoch {epoch} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, global_step, best_val_loss,
                          output_dir / f"{config.experiment_name}_best.pt")
            print(f"  ✓ New best model saved!")

    return global_step, best_val_loss


def evaluate_movie(
    model: EmpathyAnalyzerModel,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig
) -> float:
    """Evaluate movie-level IMDb prediction."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    all_imdb_preds = []
    all_imdb_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating Movies", leave=False):
            movie_batch = batch[0]

            input_ids = movie_batch["input_ids"].to(device)
            attention_mask = movie_batch["attention_mask"].to(device)
            emotion_labels = movie_batch["emotion_labels"].to(device)
            empathy_scores = movie_batch["empathy_scores"].to(device)
            style_labels = movie_batch["style_labels"].to(device)
            character_ids = movie_batch["character_ids"].to(device)
            num_characters = movie_batch["num_characters"]
            imdb_target = movie_batch["imdb_rating"].to(device)

            with autocast(enabled=config.mixed_precision):
                outputs = model(
                    input_ids, attention_mask,
                    character_ids=character_ids,
                    num_characters=num_characters,
                    return_char_profiles=True
                )

                targets = {
                    "emotion_labels": emotion_labels,
                    "empathy_scores": empathy_scores,
                    "style_labels": style_labels
                }

                loss, _ = model.get_loss(
                    outputs, targets,
                    character_profiles=outputs.get("character_profiles"),
                    imdb_target=imdb_target.unsqueeze(0)
                )

            total_loss += loss.item()
            total_batches += 1

            # IMDb prediction
            char_profiles = outputs.get("character_profiles")
            if char_profiles is not None:
                imdb_pred = model.predict_imdb(char_profiles)
                all_imdb_preds.append(imdb_pred.cpu())
                all_imdb_targets.append(imdb_target.cpu())

    if all_imdb_preds:
        imdb_preds = torch.cat(all_imdb_preds)
        imdb_targets = torch.cat(all_imdb_targets)
        imdb_mse = torch.nn.functional.mse_loss(imdb_preds, imdb_targets).item()
        imdb_mae = torch.nn.functional.l1_loss(imdb_preds, imdb_targets).item()
        print(f"  IMDb MSE: {imdb_mse:.4f}, MAE: {imdb_mae:.4f}")

    return total_loss / max(total_batches, 1)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,
    step: int,
    best_val_loss: float,
    path: Path
):
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "config": model.config
    }, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler.LambdaLR] = None
) -> Tuple[int, float]:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["step"], checkpoint["best_val_loss"]


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: TrainingConfig,
    num_training_steps: int
) -> Tuple[optim.Optimizer, optim.lr_scheduler.LambdaLR]:
    """Create optimizer with weight decay and linear warmup + cosine decay."""

    # Separate params for weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name.lower() or "embedding" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8)

    def lr_lambda(current_step: int):
        if current_step < config.warmup_steps:
            return float(current_step) / float(max(1, config.warmup_steps))
        progress = float(current_step - config.warmup_steps) / float(max(1, num_training_steps - config.warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description="Train Custom Empathy Analyzer")
    parser.add_argument("--config", type=str, help="Path to training config JSON")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--phase", type=str, choices=["1", "2", "3", "all"], default="all",
                        help="Training phase: 1=emotion+empathy, 2=+style, 3=+imdb, all=all phases")
    parser.add_argument("--epochs", type=int, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()

    # Override from CLI
    if args.epochs:
        config.max_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr

    # Setup
    set_seed(config.seed)
    device = get_device(config.device)
    print(f"Using device: {device}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(asdict(config), f, indent=2)

    # Prepare tokenizer
    tokenizer = prepare_tokenizer(config)

    # Prepare IMDb ratings
    imdb_ratings = prepare_imdb_ratings(config)

    # Create model
    model_config = config.to_model_config()
    model = create_model(model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataloaders
    print("Creating dialogue-level dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config.data_path, tokenizer, model_config,
        batch_size=config.batch_size,
        max_length=config.max_seq_len,
        num_workers=config.num_workers
    )

    print("Creating movie-level dataloaders...")
    movie_train_loader, movie_val_loader, movie_test_loader = create_movie_dataloaders(
        config.data_path, config.imdb_ratings_path, tokenizer, model_config,
        batch_size=config.movie_batch_size,
        max_length=config.max_seq_len,
        num_workers=config.num_workers
    )

    # Optimizer & scheduler
    num_training_steps = len(train_loader) * config.max_epochs // config.gradient_accumulation_steps
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, num_training_steps)

    # Mixed precision scaler
    scaler = GradScaler(enabled=config.mixed_precision)

    # Resume if requested
    global_step = 0
    best_val_loss = float('inf')
    if args.resume:
        global_step, best_val_loss = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler
        )
        print(f"Resumed from step {global_step}, best val loss: {best_val_loss:.4f}")

    # ===== PHASE 1: Emotion + Empathy =====
    if args.phase in ["1", "all"]:
        print("\n" + "="*60)
        print("PHASE 1: Emotion Classification + Empathy Regression")
        print("="*60)
        # Freeze style head
        for param in model.style_head.parameters():
            param.requires_grad = False
        for param in model.imdb_head.parameters():
            param.requires_grad = False

        global_step, best_val_loss = train_dialogue_phase(
            model, train_loader, val_loader,
            optimizer, scheduler, scaler, device, config,
            "Phase1", config.phase1_epochs, global_step, best_val_loss, output_dir
        )

    # ===== PHASE 2: + Style =====
    if args.phase in ["2", "all"]:
        print("\n" + "="*60)
        print("PHASE 2: Adding Communication Style Classification")
        print("="*60)
        # Unfreeze style head
        for param in model.style_head.parameters():
            param.requires_grad = True
        for param in model.imdb_head.parameters():
            param.requires_grad = False

        global_step, best_val_loss = train_dialogue_phase(
            model, train_loader, val_loader,
            optimizer, scheduler, scaler, device, config,
            "Phase2", config.phase2_epochs, global_step, best_val_loss, output_dir
        )

    # ===== PHASE 3: + IMDb (Movie-level) =====
    if args.phase in ["3", "all"]:
        print("\n" + "="*60)
        print("PHASE 3: Adding IMDb Rating Prediction (Movie-level)")
        print("="*60)
        # Unfreeze IMDb head
        for param in model.imdb_head.parameters():
            param.requires_grad = True

        global_step, best_val_loss = train_movie_phase(
            model, movie_train_loader, movie_val_loader,
            optimizer, scheduler, scaler, device, config,
            global_step, best_val_loss, output_dir
        )

    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)

    test_loss = evaluate_dialogue(model, test_loader, device, config)
    print(f"Test Dialogue Loss: {test_loss:.4f}")

    if args.phase in ["3", "all"]:
        movie_test_loss = evaluate_movie(model, movie_test_loader, device, config)
        print(f"Test Movie Loss: {movie_test_loss:.4f}")

    # Save final model
    final_path = output_dir / f"{config.experiment_name}_final.pt"
    save_checkpoint(model, optimizer, scheduler, global_step, best_val_loss, final_path)
    print(f"\n✓ Training complete! Final model saved to {final_path}")


if __name__ == "__main__":
    main()