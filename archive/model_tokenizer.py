"""
Custom Subword Tokenizer for Screenplay Dialogue
=================================================
Trains a BPE (Byte Pair Encoding) tokenizer on screenplay corpus.
Optimized for dialogue: handles character names, parentheticals, 
screenplay-specific vocabulary, and emotional expressions.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter
import pickle


class ScreenplayBPETokenizer:
    """
    BPE tokenizer trained specifically on screenplay dialogue.
    Handles:
    - Character names (ALL_CAPS)
    - Parentheticals (V.O.), (O.S.), (CONT'D)
    - Scene headings (INT., EXT.)
    - Emotional expressions (...), (--), (!), (?)
    - Contractions and informal speech
    """

    def __init__(
        self,
        vocab_size: int = 12000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or [
            "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>",
            "<CHAR>", "<SCENE>", "<PAREN>", "<EMOTION>"
        ]

        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Pre-compiled regex for preprocessing
        self._whitespace_re = re.compile(r'\s+')
        self._char_name_re = re.compile(r'^[A-Z]{2,}(?:\s+[A-Z]{2,})*$')
        self._scene_heading_re = re.compile(r'^(?:INT\.|EXT\.|INT/EXT\.|I/E\.)')
        self._parenthetical_re = re.compile(r'\([^)]*\)')

    def _preprocess_text(self, text: str) -> str:
        """Normalize screenplay text for training."""
        # Preserve structure markers
        text = text.replace('...', ' <ELLIPSIS> ')
        text = text.replace('--', ' <DASH> ')
        text = text.replace('?', ' ? ')
        text = text.replace('!', ' ! ')
        text = text.replace(',', ' , ')
        text = text.replace('.', ' . ')
        text = text.replace(':', ' : ')
        text = text.replace(';', ' ; ')
        text = text.replace('(', ' ( ')
        text = text.replace(')', ' ) ')
        text = text.replace('[', ' [ ')
        text = text.replace(']', ' ] ')
        text = text.replace('{', ' { ')
        text = text.replace('}', ' } ')

        # Normalize whitespace
        text = self._whitespace_re.sub(' ', text)
        return text.strip()

    def _get_word_frequencies(self, texts: List[str]) -> Counter:
        """Get word-level frequencies from preprocessed texts."""
        word_freqs = Counter()
        for text in texts:
            processed = self._preprocess_text(text)
            words = processed.split()
            word_freqs.update(words)
        return word_freqs

    def _compute_pair_frequencies(self, word_freqs: Counter) -> Counter:
        """Compute frequencies of adjacent symbol pairs."""
        pair_freqs = Counter()
        for word, freq in word_freqs.items():
            symbols = word.split()
            if len(symbols) < 2:
                continue
            for i in range(len(symbols) - 1):
                pair_freqs[(symbols[i], symbols[i + 1])] += freq
        return pair_freqs

    def _merge_pair(self, word_freqs: Counter, pair: Tuple[str, str]) -> Counter:
        """Merge a pair of symbols in all words."""
        new_word_freqs = Counter()
        bigram = ' '.join(pair)
        merged = ''.join(pair)

        for word, freq in word_freqs.items():
            if bigram in word:
                new_word = word.replace(bigram, merged)
                new_word_freqs[new_word] = freq
            else:
                new_word_freqs[word] = freq
        return new_word_freqs

    def train(self, texts: List[str], verbose: bool = True) -> 'ScreenplayBPETokenizer':
        """
        Train BPE tokenizer on screenplay texts.

        Args:
            texts: List of raw dialogue strings
            verbose: Print progress

        Returns:
            self (for chaining)
        """
        if verbose:
            print(f"Training BPE tokenizer on {len(texts)} texts...")
            print(f"Target vocab size: {self.vocab_size}")

        # Initialize vocabulary with characters + special tokens
        vocab = set()
        for text in texts:
            processed = self._preprocess_text(text)
            for ch in processed:
                if ch != ' ':
                    vocab.add(ch)

        # Add special tokens
        for token in self.special_tokens:
            vocab.add(token)

        # Word frequencies
        word_freqs = self._get_word_frequencies(texts)

        # Filter by min_frequency
        word_freqs = Counter({w: f for w, f in word_freqs.items() if f >= self.min_frequency})

        if verbose:
            print(f"Unique characters: {len(vocab)}")
            print(f"Unique words (freq >= {self.min_frequency}): {len(word_freqs)}")

        # BPE merges
        num_merges = self.vocab_size - len(vocab)
        num_merges = max(0, num_merges)

        for i in range(num_merges):
            pair_freqs = self._compute_pair_frequencies(word_freqs)
            if not pair_freqs:
                break

            best_pair = pair_freqs.most_common(1)[0][0]
            self.merges.append(best_pair)
            word_freqs = self._merge_pair(word_freqs, best_pair)
            vocab.add(''.join(best_pair))

            if verbose and (i + 1) % 1000 == 0:
                print(f"  Merge {i + 1}/{num_merges}: {best_pair} -> '{''.join(best_pair)}'")

        # Build final vocab mappings
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab))}
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}

        # Ensure special tokens have fixed IDs
        for i, token in enumerate(self.special_tokens):
            if token in self.vocab:
                # Move to front
                old_idx = self.vocab[token]
                if old_idx != i:
                    # Swap
                    other_token = self.inv_vocab[i]
                    self.vocab[token] = i
                    self.vocab[other_token] = old_idx
                    self.inv_vocab[i] = token
                    self.inv_vocab[old_idx] = other_token

        self.token_to_id = self.vocab.copy()
        self.id_to_token = self.inv_vocab.copy()

        if verbose:
            print(f"Training complete. Final vocab size: {len(self.vocab)}")
            print(f"Special token IDs: { {t: self.vocab[t] for t in self.special_tokens} }")

        return self

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        processed = self._preprocess_text(text)
        words = processed.split()

        token_ids = []
        if add_special_tokens:
            token_ids.append(self.token_to_id["<BOS>"])

        for word in words:
            # Try to tokenize word using learned merges
            word_tokens = self._tokenize_word(word)
            token_ids.extend(word_tokens)

        if add_special_tokens:
            token_ids.append(self.token_to_id["<EOS>"])

        return token_ids

    def _tokenize_word(self, word: str) -> List[int]:
        """Apply BPE merges to a single word."""
        if word in self.token_to_id:
            return [self.token_to_id[word]]

        # Start with character-level
        tokens = list(word)
        token_ids = []

        # Apply merges in order
        for merge in self.merges:
            merged = ''.join(merge)
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == merge[0] and tokens[i + 1] == merge[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Convert to IDs
        for token in tokens:
            token_ids.append(self.token_to_id.get(token, self.token_to_id["<UNK>"]))

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for tid in token_ids:
            token = self.id_to_token.get(tid, "<UNK>")
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)

        # Join and clean up
        text = ''.join(tokens)
        text = text.replace('<ELLIPSIS>', '...')
        text = text.replace('<DASH>', '--')
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def save(self, path: str) -> None:
        """Save tokenizer to disk."""
        data = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "merges": self.merges,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'ScreenplayBPETokenizer':
        """Load tokenizer from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"],
            min_frequency=data["min_frequency"],
            special_tokens=data["special_tokens"]
        )
        tokenizer.vocab = data["vocab"]
        tokenizer.merges = data["merges"]
        tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.token_to_id = tokenizer.vocab.copy()
        tokenizer.id_to_token = tokenizer.inv_vocab.copy()
        return tokenizer

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def token_to_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id["<UNK>"])

    def id_to_token(self, idx: int) -> str:
        return self.id_to_token.get(idx, "<UNK>")


def train_tokenizer_from_screenplays(
    screenplay_dir: str,
    output_path: str,
    vocab_size: int = 12000
) -> ScreenplayBPETokenizer:
    """
    Convenience function to train tokenizer on all screenplays in a directory.
    """
    screenplay_dir = Path(screenplay_dir)
    texts = []

    for txt_file in screenplay_dir.glob("*.txt"):
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract dialogue lines (simple extraction for tokenizer training)
        # This is a simplified version - the full parser is in app.py
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('INT.', 'EXT.', 'FADE', 'CUT TO', 'DISSOLVE')):
                # Likely dialogue or character name
                if ':' in line or line.isupper():
                    texts.append(line)

    print(f"Extracted {len(texts)} lines for tokenizer training")

    tokenizer = ScreenplayBPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts)
    tokenizer.save(output_path)

    return tokenizer


if __name__ == "__main__":
    # Quick test
    tokenizer = ScreenplayBPETokenizer(vocab_size=1000)
    sample_texts = [
        "BATEMAN: you'll notice that my friends and i all look similar",
        "MCDERMOTT: what do you mean? that's not paul owen!",
        "VAN PATTEN: ...well, i think you're right actually",
        "INT. RESTAURANT - NIGHT",
        "BATEMAN V/O: my name is patrick bateman"
    ]
    tokenizer.train(sample_texts, verbose=True)

    # Test encode/decode
    test = "BATEMAN: you think so? you'll look like you consciously worked for the look"
    encoded = tokenizer.encode(test)
    decoded = tokenizer.decode(encoded)
    print(f"\nOriginal: {test}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")