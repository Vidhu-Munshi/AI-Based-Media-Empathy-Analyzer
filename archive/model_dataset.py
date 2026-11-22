"""
PyTorch Dataset for Screenplay Empathy Analysis
================================================
Handles labeled data (JSONL format) and converts to model-ready tensors.
Supports multi-task learning: emotion, empathy, style, IMDb rating.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import random
from collections import defaultdict

from model.tokenizer import ScreenplayBPETokenizer
from model.custom_model import (
    ModelConfig, EMOTION_LABELS, STYLE_LABELS,
    EMOTION_TO_ID, STYLE_TO_ID
)


@dataclass
class DialogueSample:
    """Single dialogue line with all annotations."""
    dialogue: str
    character: str
    movie: str
    scene: str
    line_id: str
    emotions: List[str]           # Multi-label from EMOTION_LABELS
    empathy_score: float          # [-3, +3]
    style: str                    # Single from STYLE_LABELS
    confidence: float = 1.0
    notes: str = ""


@dataclass
class MovieSample:
    """Aggregated movie-level sample for IMDb rating prediction."""
    movie: str
    imdb_rating: float            # Ground truth IMDb rating [1, 10]
    dialogues: List[DialogueSample]
    character_order: List[str]    # Consistent character ordering


class ScreenplayDataset(Dataset):
    """
    Dataset for dialogue-level tasks (emotion, empathy, style).
    Each item = one dialogue line.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: ScreenplayBPETokenizer,
        config: ModelConfig,
        max_length: int = 256,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        self.split = split

        # Load all samples
        all_samples = self._load_jsonl(data_path)

        # Split
        random.seed(seed)
        random.shuffle(all_samples)
        n = len(all_samples)
        train_end = int(n * split_ratios[0])
        val_end = train_end + int(n * split_ratios[1])

        if split == "train":
            self.samples = all_samples[:train_end]
        elif split == "val":
            self.samples = all_samples[train_end:val_end]
        elif split == "test":
            self.samples = all_samples[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")

        print(f"{split.capitalize()} dataset: {len(self.samples)} samples")

    def _load_jsonl(self, path: str) -> List[DialogueSample]:
        """Load labeled data from JSONL file."""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sample = DialogueSample(
                        dialogue=data["dialogue"],
                        character=data["character"],
                        movie=data["movie"],
                        scene=data["scene"],
                        line_id=data.get("line_id", f"line_{line_num}"),
                        emotions=data["labels"]["emotions"],
                        empathy_score=data["labels"]["empathy_score"],
                        style=data["labels"]["style"],
                        confidence=data.get("meta", {}).get("confidence", 1.0),
                        notes=data.get("meta", {}).get("notes", "")
                    )
                    # Validate
                    self._validate_sample(sample)
                    samples.append(sample)
                except Exception as e:
                    print(f"Warning: Skipping line {line_num}: {e}")
        return samples

    def _validate_sample(self, sample: DialogueSample) -> None:
        """Validate sample labels."""
        # Check emotions
        for emo in sample.emotions:
            if emo not in EMOTION_TO_ID:
                raise ValueError(f"Unknown emotion: {emo}")

        # Check style
        if sample.style not in STYLE_TO_ID:
            raise ValueError(f"Unknown style: {sample.style}")

        # Check empathy score range
        if not (-3.0 <= sample.empathy_score <= 3.0):
            raise ValueError(f"Empathy score out of range: {sample.empathy_score}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Tokenize
        token_ids = self.tokenizer.encode(sample.dialogue, add_special_tokens=True)
        token_ids = token_ids[:self.max_length]

        # Pad
        attention_mask = [1] * len(token_ids)
        pad_length = self.max_length - len(token_ids)
        if pad_length > 0:
            token_ids += [self.config.pad_token_id] * pad_length
            attention_mask += [0] * pad_length

        # Emotion labels (multi-hot)
        emotion_labels = torch.zeros(self.config.num_emotions, dtype=torch.float)
        for emo in sample.emotions:
            emotion_labels[EMOTION_TO_ID[emo]] = 1.0

        # Empathy score
        empathy_score = torch.tensor(sample.empathy_score, dtype=torch.float)

        # Style label (single class)
        style_label = torch.tensor(STYLE_TO_ID[sample.style], dtype=torch.long)

        # Confidence weight
        confidence = torch.tensor(sample.confidence, dtype=torch.float)

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "emotion_labels": emotion_labels,
            "empathy_score": empathy_score,
            "style_label": style_label,
            "confidence": confidence,
            # Metadata for aggregation
            "character": sample.character,
            "movie": sample.movie,
            "line_id": sample.line_id
        }


class MovieLevelDataset(Dataset):
    """
    Dataset for movie-level IMDb rating prediction.
    Each item = one movie with all its dialogues aggregated.
    """

    def __init__(
        self,
        data_path: str,
        imdb_ratings_path: str,  # JSON: {"movie_name": rating}
        tokenizer: ScreenplayBPETokenizer,
        config: ModelConfig,
        max_length: int = 256,
        max_dialogues_per_movie: int = 500,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        self.max_dialogues_per_movie = max_dialogues_per_movie
        self.split = split

        # Load IMDb ratings
        with open(imdb_ratings_path, 'r') as f:
            self.imdb_ratings = json.load(f)

        # Load and group by movie
        all_samples = self._load_jsonl(data_path)
        self.movie_samples = self._group_by_movie(all_samples)

        # Split movies
        movie_names = list(self.movie_samples.keys())
        random.seed(seed)
        random.shuffle(movie_names)
        n = len(movie_names)
        train_end = int(n * split_ratios[0])
        val_end = train_end + int(n * split_ratios[1])

        if split == "train":
            split_movies = movie_names[:train_end]
        elif split == "val":
            split_movies = movie_names[train_end:val_end]
        elif split == "test":
            split_movies = movie_names[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.movies = [self.movie_samples[m] for m in split_movies if m in self.imdb_ratings]
        print(f"{split.capitalize()} movie dataset: {len(self.movies)} movies")

    def _load_jsonl(self, path: str) -> List[DialogueSample]:
        """Same as ScreenplayDataset."""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sample = DialogueSample(
                        dialogue=data["dialogue"],
                        character=data["character"],
                        movie=data["movie"],
                        scene=data["scene"],
                        line_id=data.get("line_id", f"line_{line_num}"),
                        emotions=data["labels"]["emotions"],
                        empathy_score=data["labels"]["empathy_score"],
                        style=data["labels"]["style"],
                        confidence=data.get("meta", {}).get("confidence", 1.0),
                        notes=data.get("meta", {}).get("notes", "")
                    )
                    samples.append(sample)
                except Exception as e:
                    print(f"Warning: Skipping line {line_num}: {e}")
        return samples

    def _group_by_movie(self, samples: List[DialogueSample]) -> Dict[str, MovieSample]:
        """Group dialogue samples by movie."""
        movie_dict = defaultdict(list)
        for sample in samples:
            movie_dict[sample.movie].append(sample)

        movie_samples = {}
        for movie, dialogues in movie_dict.items():
            # Get consistent character ordering
            characters = list(set(d.character for d in dialogues))
            characters.sort()  # Deterministic order

            movie_samples[movie] = MovieSample(
                movie=movie,
                imdb_rating=self.imdb_ratings.get(movie, 5.0),  # Default if missing
                dialogues=dialogues,
                character_order=characters
            )
        return movie_samples

    def __len__(self) -> int:
        return len(self.movies)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        movie_sample = self.movies[idx]

        # Sample dialogues (limit for memory)
        dialogues = movie_sample.dialogues
        if len(dialogues) > self.max_dialogues_per_movie:
            dialogues = random.sample(dialogues, self.max_dialogues_per_movie)

        # Tokenize all dialogues
        all_input_ids = []
        all_attention_masks = []
        all_emotion_labels = []
        all_empathy_scores = []
        all_style_labels = []
        character_ids = []

        char_to_idx = {char: i for i, char in enumerate(movie_sample.character_order)}

        for d in dialogues:
            token_ids = self.tokenizer.encode(d.dialogue, add_special_tokens=True)
            token_ids = token_ids[:self.max_length]

            attention_mask = [1] * len(token_ids)
            pad_length = self.max_length - len(token_ids)
            if pad_length > 0:
                token_ids += [self.config.pad_token_id] * pad_length
                attention_mask += [0] * pad_length

            all_input_ids.append(token_ids)
            all_attention_masks.append(attention_mask)

            # Emotion labels
            emotion_labels = torch.zeros(self.config.num_emotions, dtype=torch.float)
            for emo in d.emotions:
                emotion_labels[EMOTION_TO_ID[emo]] = 1.0
            all_emotion_labels.append(emotion_labels)

            # Empathy score
            all_empathy_scores.append(torch.tensor(d.empathy_score, dtype=torch.float))

            # Style label
            all_style_labels.append(torch.tensor(STYLE_TO_ID[d.style], dtype=torch.long))

            # Character index
            character_ids.append(char_to_idx[d.character])

        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),           # (num_dialogues, max_length)
            "attention_mask": torch.tensor(all_attention_masks, dtype=torch.long),
            "emotion_labels": torch.stack(all_emotion_labels),                    # (num_dialogues, num_emotions)
            "empathy_scores": torch.stack(all_empathy_scores),                    # (num_dialogues,)
            "style_labels": torch.stack(all_style_labels),                        # (num_dialogues,)
            "character_ids": torch.tensor(character_ids, dtype=torch.long),       # (num_dialogues,)
            "num_characters": len(movie_sample.character_order),
            "imdb_rating": torch.tensor(movie_sample.imdb_rating, dtype=torch.float),
            "movie": movie_sample.movie
        }


def collate_dialogue_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for ScreenplayDataset (dialogue-level)."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "emotion_labels": torch.stack([b["emotion_labels"] for b in batch]),
        "empathy_scores": torch.stack([b["empathy_score"] for b in batch]),
        "style_labels": torch.stack([b["style_label"] for b in batch]),
        "confidence": torch.stack([b["confidence"] for b in batch]),
        "character": [b["character"] for b in batch],
        "movie": [b["movie"] for b in batch],
        "line_id": [b["line_id"] for b in batch]
    }


def collate_movie_batch(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for MovieLevelDataset (variable number of dialogues per movie)."""
    # Since each movie has different number of dialogues, we can't stack directly
    # Return as list of dicts, handle in training loop
    return batch


def create_dataloaders(
    data_path: str,
    tokenizer: ScreenplayBPETokenizer,
    config: ModelConfig,
    batch_size: int = 32,
    max_length: int = 256,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders for dialogue-level tasks."""
    train_dataset = ScreenplayDataset(data_path, tokenizer, config, max_length, "train")
    val_dataset = ScreenplayDataset(data_path, tokenizer, config, max_length, "val")
    test_dataset = ScreenplayDataset(data_path, tokenizer, config, max_length, "test")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_dialogue_batch, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_dialogue_batch, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_dialogue_batch, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def create_movie_dataloaders(
    data_path: str,
    imdb_ratings_path: str,
    tokenizer: ScreenplayBPETokenizer,
    config: ModelConfig,
    batch_size: int = 1,  # One movie per batch (variable size)
    max_length: int = 256,
    max_dialogues_per_movie: int = 500,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders for movie-level IMDb prediction."""
    train_dataset = MovieLevelDataset(
        data_path, imdb_ratings_path, tokenizer, config, max_length,
        max_dialogues_per_movie, "train"
    )
    val_dataset = MovieLevelDataset(
        data_path, imdb_ratings_path, tokenizer, config, max_length,
        max_dialogues_per_movie, "val"
    )
    test_dataset = MovieLevelDataset(
        data_path, imdb_ratings_path, tokenizer, config, max_length,
        max_dialogues_per_movie, "test"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_movie_batch, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_movie_batch, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_movie_batch, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


# Example usage and testing
if __name__ == "__main__":
    from model.custom_model import ModelConfig
    from model.tokenizer import ScreenplayBPETokenizer

    # Quick test with dummy data
    config = ModelConfig(vocab_size=1000)
    tokenizer = ScreenplayBPETokenizer(vocab_size=1000)

    sample_texts = [
        "BATEMAN: you'll notice that my friends and i all look similar",
        "MCDERMOTT: what do you mean? that's not paul owen!",
        "VAN PATTEN: ...well, i think you're right actually"
    ]
    tokenizer.train(sample_texts, verbose=False)

    # Create dummy JSONL
    dummy_data = [
        {
            "dialogue": "you'll notice that my friends and i all look similar",
            "character": "BATEMAN",
            "movie": "AMERICAN_PSYCHO",
            "scene": "INT. RESTAURANT - NIGHT",
            "line_id": "AP_001",
            "labels": {
                "emotions": ["condescension", "pride"],
                "empathy_score": -1.2,
                "style": "condescending"
            },
            "meta": {"confidence": 0.9, "notes": ""}
        },
        {
            "dialogue": "what do you mean? that's not paul owen!",
            "character": "MCDERMOTT",
            "movie": "AMERICAN_PSYCHO",
            "scene": "INT. RESTAURANT - NIGHT",
            "line_id": "AP_002",
            "labels": {
                "emotions": ["surprise", "anger"],
                "empathy_score": -0.5,
                "style": "inquisitive"
            },
            "meta": {"confidence": 0.8, "notes": ""}
        }
    ]

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in dummy_data:
            f.write(json.dumps(item) + '\n')
        temp_path = f.name

    # Test dataset
    dataset = ScreenplayDataset(temp_path, tokenizer, config, max_length=64, split="train")
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"emotion_labels shape: {sample['emotion_labels'].shape}")
    print(f"empathy_score: {sample['empathy_score']}")
    print(f"style_label: {sample['style_label']}")

    # Test dataloader
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_dialogue_batch)
    batch = next(iter(loader))
    print(f"\nBatch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch emotion_labels shape: {batch['emotion_labels'].shape}")