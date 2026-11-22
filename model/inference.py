"""Reusable, checkpoint-free screenplay parsing and feature helpers."""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

from model.kruti_to_unicode import is_kruti_dev, kruti_to_unicode


def clean_dialogue_text(text: str) -> str:
    """Clean dialogue text while preserving both English and Devanagari/Hindi characters."""
    text = re.sub(r"\(.*?\)", "", text).lower()
    text = text.replace("…", "...").replace("—", "--").replace("–", "--")
    # Preserve English letters, digits, Devanagari Unicode (\u0900-\u097f), and standard punctuation
    text = re.sub(r"[^a-zA-Z0-9\u0900-\u097f\s\.,!\?'\"-]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_metadata(line: str) -> bool:
    clean = line.strip()
    if not clean:
        return False
    # Page numbers
    if re.match(r"^(?:\d{1,4}\.?|\[?\d{1,4}\]?|\(\d{1,4}\))$", clean):
        return True
    # Copyright / license / web source / Gutenberg
    if re.search(r"(copyright|©|all rights reserved|project gutenberg|electronic version|transcribed by)", clean, re.IGNORECASE):
        return True
    # URLs and web domains
    if re.search(r"https?://|www\.|(?:\w+\.(?:edu|org|com|net|gov))", clean, re.IGNORECASE):
        return True
    # Book front-matter / biographical headers: e.g. "Published:", "Source:", "About <Author>:", "children:", "Author:"
    if re.match(r"^(?:published|source|category|categorie|genre|author|title|isbn|ebook|edition|editor|translator|biography|about\s+\w+|children|born|died|spouse|parents|family)\s*[:\s]", clean, re.IGNORECASE):
        return True
    return False


# Words that look like uppercase character headings but are actually
# scene directions, transitions, or common English words.
_FALSE_CHARACTER_NAMES = {
    'THE', 'AND', 'BUT', 'FOR', 'WITH', 'FROM', 'INTO', 'ONTO', 'OVER',
    'HIS', 'HER', 'HIM', 'SHE', 'HE', 'THEY', 'THEM', 'WE', 'YOU',
    'THIS', 'THAT', 'THEN', 'THAN', 'WHEN', 'WHERE', 'WHAT', 'WHO',
    'LATER', 'MUCH LATER', 'NEXT', 'MEANWHILE', 'SUDDENLY', 'MORNING',
    'EVENING', 'NIGHT', 'DAY', 'DAWN', 'DUSK', 'NOON', 'AFTERNOON',
    'CONTINUED', 'CONTINUOUS', 'RESUME', 'END', 'BEGIN',
    'CUTTO', 'CUT TO', 'SMASH CUT', 'MATCH CUT', 'JUMP CUT',
    'FADE', 'DISSOLVE', 'WIPE', 'INTERCUT', 'FLASHBACK', 'MONTAGE',
    'TITLE', 'SUBTITLE', 'SUPER', 'CLOSE ON', 'ANGLE ON', 'BACK TO',
    'SAME', 'MORE', 'ALSO', 'STILL', 'JUST', 'ONLY', 'VERY', 'MUCH',
    'STYLE', 'FLAW', 'MUTTERING', 'SILENCE', 'BEAT', 'PAUSE',
}

# Common English sentence/prose words (pronouns, auxiliary verbs, conjunctions)
# that should never be identified as character names (e.g. "I fear it is", "There art thou happy", "Depart again")
_SENTENCE_PROSE_WORDS = {
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'its', 'our', 'their', 'thou', 'thee', 'thy', 'thine', 'ye',
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'art', 'hast', 'hath', 'doth', 'doest',
    'have', 'has', 'had', 'do', 'does', 'did', 'shall', 'will', 'would', 'could', 'should',
    'may', 'might', 'must', 'can', 'cannot', 'wilt', 'shalt',
    'there', 'here', 'where', 'when', 'why', 'how', 'what', 'who', 'whom', 'which',
    'this', 'that', 'these', 'those', 'again', 'then', 'than', 'so', 'as', 'if', 'but',
    'not', 'no', 'nor', 'too', 'very', 'fear', 'happy', 'depart', 'come', 'go', 'see',
    'say', 'said', 'tell', 'told', 'speak', 'spoke', 'look', 'looks', 'looked', 'think', 'thought',
    'marriage', 'employment', 'tomorrow', 'to-morrow', 'yesterday', 'night', 'morning', 'evening',
    'day', 'noon', 'midnight', 'hour', 'hours', 'clock', 'time', 'life', 'death', 'world', 'earth'
}

# Prepositions and greetings that often appear before colons in poetry/dialogue but are NOT character names
_PREPOSITIONS_AND_GREETINGS = {
    'about', 'against', 'by', 'in', 'on', 'at', 'to', 'from', 'with', 'into', 'onto',
    'upon', 'under', 'above', 'between', 'before', 'after', 'for', 'of', 'toward',
    'towards', 'through', 'throughout', 'during', 'without', 'within',
    'farewell', 'good night', 'good morning', 'good day', 'goodbye', 'alas', 'welcome',
    'prithee', 'hark', 'behold', 'amen', 'marry', 'peace', 'ho', 'hallo', 'hail'
}

# Metadata keywords that appear in book front-matters or prefaces
_BOOK_METADATA_TERMS = {
    'published', 'source', 'author', 'title', 'category', 'categorie', 'genre', 'children',
    'born', 'died', 'spouse', 'parents', 'family', 'gutenberg', 'ebook', 'edition', 'editor',
    'translator', 'website', 'url', 'http', 'https', 'summary', 'synopsis', 'preface',
    'introduction', 'biography', 'notes', 'note', 'act', 'scene', 'chapter', 'verse', 'line'
}


def _is_valid_character_name(candidate: str) -> bool:
    """Filter out false positive character headings."""
    if not candidate or len(candidate) < 2:
        return False
    candidate_clean = candidate.strip()
    clean_upper = candidate_clean.upper()

    if clean_upper in _FALSE_CHARACTER_NAMES:
        return False

    # English validation
    if re.search(r'[A-Za-z]', candidate_clean):
        # Reject punctuation commonly in dialogue clauses (commas, semicolons, quotes)
        if any(p in candidate_clean for p in (',', ';', '"', '!', '?')):
            return False

        # If it's ALL CAPS (e.g. ROMEO, JULIET, CAPULET, LADY CAPULET, FRIAR LAURENCE)
        if candidate_clean.isupper():
            words = candidate_clean.split()
            if len(words) > 4:
                return False
            if clean_upper.startswith('THE ') or clean_upper.startswith('A '):
                return False
            if 'IN THE' in clean_upper or 'OF THE' in clean_upper or 'AT THE' in clean_upper:
                return False
            return True

        # If NOT all caps (mixed case / Title Case like "Romeo", "Friar Laurence")
        words = candidate_clean.split()
        # In theatrical scripts, mixed-case character headings are at most 2 words (e.g. "Romeo", "Juliet", "Friar Laurence")
        if len(words) > 2:
            return False

        # Every word must start with an uppercase letter (Title Case) - no lowercase words like 'up', 'the', 'none'
        if any(not w[0].isupper() for w in words):
            return False

        # Author / book metadata
        if candidate_clean.lower() in ('shakespeare', 'william shakespeare', 'author', 'editor', 'published', 'source', 'translator'):
            return False

        lower_words = [w.strip(".,!?;:\"'-").lower() for w in words]
        first_word = lower_words[0] if lower_words else ""

        if first_word in _PREPOSITIONS_AND_GREETINGS or candidate_clean.lower() in _PREPOSITIONS_AND_GREETINGS:
            return False
        if any(w in _BOOK_METADATA_TERMS for w in lower_words):
            return False
        if any(w in _SENTENCE_PROSE_WORDS for w in lower_words):
            return False

        return True

    # Devanagari / Hindi characters
    words = candidate_clean.split()
    if len(words) > 4:
        return False
    return True


def parse_screenplay_text(content: str, movie_name: str = "unknown") -> pd.DataFrame:
    """Parse common uppercase English or Hindi-style screenplay layouts into dialogue rows."""
    if not content:
        return pd.DataFrame(columns=["Movie", "Scene", "Character", "Dialogue"])

    # Auto-convert Kruti Dev if legacy Hindi font is detected
    if is_kruti_dev(content):
        content = kruti_to_unicode(content)

    rows, scene, character, dialogue = [], "UNKNOWN_SCENE", None, []

    def flush() -> None:
        nonlocal dialogue
        if character and dialogue:
            text = clean_dialogue_text(" ".join(dialogue))
            if text:
                rows.append({"Movie": movie_name, "Scene": scene, "Character": character, "Dialogue": text})
        dialogue = []

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            flush(); character = None; continue
        if _is_metadata(line):
            continue

        # Speaker format: e.g. "त्रिभुवन : संवाद" or "ROMEO: O speak again" or "हुस्ना % संवाद"
        speaker_match = re.match(r'^([\u0900-\u097fA-Za-z0-9०-९\s\/\-]{1,35})\s*[:：%]\s*(.*)$', line)
        if speaker_match:
            candidate = speaker_match.group(1).strip()
            inline = speaker_match.group(2).strip()
            if _is_valid_character_name(candidate):
                flush()
                character = candidate
                if inline:
                    dialogue.append(inline)
                continue
            else:
                # If not a valid character name (e.g. "There art thou happy: ..."), it is dialogue!
                if character:
                    dialogue.append(line)
                continue

        # Scene headings
        if re.match(r"^(?:INT\.|EXT\.|INT/EXT\.|INT |EXT |I/E |पहला दृश्य|दूसरा दृश्य|तीसरा दृश्य|चौथा दृश्य|पाँचवाँ दृश्य|छठा दृश्य|सातवाँ दृश्य|आठवाँ दृश्य|नौवाँ दृश्य|दसवाँ दृश्य|दृश्य).*", line, re.IGNORECASE):
            flush(); scene = line; character = None; continue

        # Transitions
        if re.match(r".*(?:CUT TO:|FADE IN:|FADE OUT\.|DISSOLVE TO:)$", line, re.IGNORECASE):
            flush(); character = None; continue

        # English Character Heading
        is_upper = line.isupper() and not re.search(r'[a-z]', line)
        has_sentence_punctuation = any(mark in line for mark in ("?", "!", ",")) or line.endswith(".")
        if is_upper and len(line.split()) <= 4 and not has_sentence_punctuation:
            flush()
            candidate = re.sub(r"[^A-Z0-9\s\-/]", "", re.sub(r"\(.*?\)", "", line)).strip()
            character = candidate if _is_valid_character_name(candidate) else None
            continue

        if character:
            dialogue.append(line)

    flush()
    return pd.DataFrame(rows, columns=["Movie", "Scene", "Character", "Dialogue"])


def parse_screenplay_file(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    return parse_screenplay_text(file_path.read_text(encoding="utf-8", errors="ignore"), file_path.stem)


def extract_lexical_features(text: str) -> dict[str, float | int]:
    words = re.findall(r"\b[\w']+\b", text.lower())
    hesitation_markers = re.findall(r"\.\.\.|--|\b(?:um+|uh+|erm+|hmm+)\b", text.lower())
    return {
        "word_count": len(words),
        "lexical_diversity_ttr": len(set(words)) / len(words) if words else 0.0,
        "hesitation_markers": len(hesitation_markers),
        "hesitation_rate_per_100_words": len(hesitation_markers) / len(words) * 100 if words else 0.0,
        "is_question": int("?" in text),
    }
