import os
import re
import tempfile
import warnings
import json
from io import BytesIO
from pathlib import Path
import subprocess
from urllib import error as urlerror
from urllib import request as urlrequest
from typing import List, Dict, Optional

# Set local cache directories
project_root = os.path.abspath(os.path.dirname(__file__))
hf_cache_dir = os.path.join(project_root, "cache", "huggingface")
nltk_cache_dir = os.path.join(project_root, "cache", "nltk")

os.environ["HF_HOME"] = hf_cache_dir
# Some Windows installations cannot write Hugging Face Xet's auxiliary log/cache
# files.  Falling back to the standard HTTP downloader prevents partial model
# snapshots (for example, a tokenizer file without the model weights).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
ZERO_SHOT_MODEL_NAME = "facebook/bart-large-mnli"
MULTILINGUAL_SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MULTILINGUAL_NLI_MODEL_NAME = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
LOCAL_EMOTION_MODEL_DIR = os.path.join(hf_cache_dir, "j-hartmann_emotion_english_distilroberta_base")
LOCAL_ZERO_SHOT_MODEL_DIR = os.path.join(hf_cache_dir, "facebook_bart_large_mnli")
LOCAL_MULTILINGUAL_SENTIMENT_MODEL_DIR = os.path.join(hf_cache_dir, "cardiffnlp_twitter_xlm_roberta_base_sentiment")
LOCAL_MULTILINGUAL_NLI_MODEL_DIR = os.path.join(hf_cache_dir, "multilingual_minilmv2_l6_mnli_xnli")

from model.kruti_to_unicode import is_kruti_dev, kruti_to_unicode
from model.content_advisory import compute_parents_guide


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
# Prioritize local nltk cache directory
os.makedirs(nltk_cache_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_cache_dir)

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
import importlib
_transformers = importlib.import_module("transformers")
pipeline = _transformers.pipeline
import torch

warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE CONFIGURATION & CACHING
# ==========================================
st.set_page_config(page_title="Media Empathy Analyzer", layout="wide")

st.markdown("""
<style>
    .stApp { background: #f6f8fc; }
    [data-testid="stHeader"] { background: rgba(246, 248, 252, 0.85); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111b38 0%, #182852 100%);
    }
    [data-testid="stSidebar"] * { color: #eef4ff !important; }
    [data-testid="stSidebar"] .stSlider > div > div > div { color: #74d7ff !important; }
    .hero {
        padding: 2.2rem 2.4rem;
        border-radius: 22px;
        color: #ffffff;
        background: linear-gradient(115deg, #111b38 0%, #254d91 58%, #347d98 100%);
        margin: 0.4rem 0 1.5rem 0;
        box-shadow: 0 14px 32px rgba(27, 58, 114, 0.20);
    }
    .hero h1 { margin: 0; font-size: 2.25rem; letter-spacing: -0.03em; }
    .hero p { margin: 0.55rem 0 0; color: #dceaff; font-size: 1.04rem; }
    .hero span {
        display: inline-block; margin-top: 1rem; padding: 0.3rem 0.75rem;
        border: 1px solid rgba(255,255,255,.32); border-radius: 999px;
        background: rgba(255,255,255,.12); font-size: .82rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e9f5;
        border-radius: 16px;
        padding: 1rem 1.15rem;
        box-shadow: 0 4px 14px rgba(20, 42, 86, 0.07);
    }
    div[data-testid="stMetricLabel"] { color: #52637e; }
    div[data-testid="stMetricValue"] { color: #172b55; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid #dae3f0; }
    .stTabs [data-baseweb="tab"] {
        height: 46px; padding: 0 1rem; border-radius: 10px 10px 0 0;
        color: #586a86; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background: #e8f3ff; color: #145da0; }
    .stButton > button, .stDownloadButton > button {
        border: 0; border-radius: 10px; font-weight: 650;
        background: linear-gradient(100deg, #1e5ca8, #267d9b); color: white;
        padding: 0.55rem 1rem;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { color: white; border: 0; opacity: .92; }
    div[data-testid="stDataFrame"] { border: 1px solid #e1e8f2; border-radius: 12px; overflow: hidden; }
    .section-note { color: #5e6e88; background: #eaf4fb; border-left: 4px solid #2e88ad; padding: .75rem 1rem; border-radius: 8px; }
    .rating-panel {
        color: #fff; border-radius: 18px; padding: 1.25rem 1.4rem;
        background: linear-gradient(125deg, #ee8d21, #d35432);
        box-shadow: 0 9px 22px rgba(201, 94, 37, .22); min-height: 150px;
    }
    .rating-panel .rating-label { opacity: .9; font-size: .9rem; font-weight: 600; }
    .rating-panel .rating-value { font-size: 1.45rem; line-height: 1.2; font-weight: 750; margin: .55rem 0; }
    .rating-panel .rating-note { font-size: .8rem; opacity: .92; }
    .auto-card { background: rgba(255,255,255,.1); padding: .85rem; border-radius: 10px; margin-top: .6rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading ML Models into memory (this happens once)...")
def ensure_local_hf_model(model_name: str, local_dir: str, use_fast: bool = True):
    """Download and save a HuggingFace model locally for offline reuse."""
    if os.path.isdir(local_dir) and os.path.exists(os.path.join(local_dir, "config.json")) and any(
        os.path.exists(os.path.join(local_dir, fname)) for fname in ["pytorch_model.bin", "model.safetensors"]
    ):
        return local_dir

    os.makedirs(local_dir, exist_ok=True)
    tokenizer = _transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache_dir, use_fast=use_fast)
    model = _transformers.AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=hf_cache_dir)
    tokenizer.save_pretrained(local_dir)
    model.save_pretrained(local_dir)
    return local_dir


def load_models():
    """Downloads NLTK resources and loads HuggingFace pipelines from local cached directories."""
    # NLTK setup
    for resource in ['vader_lexicon', 'punkt', 'punkt_tab']:
        try:
            if resource == 'vader_lexicon':
                nltk.data.find(f'sentiment/{resource}.zip')
            else:
                nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, download_dir=nltk_cache_dir, quiet=True)
            
    sia = SentimentIntensityAnalyzer()
    
    # Device configuration (GPU if available)
    device = 0 if torch.cuda.is_available() else -1

    emotion_dir = ensure_local_hf_model(EMOTION_MODEL_NAME, LOCAL_EMOTION_MODEL_DIR)
    zero_shot_dir = ensure_local_hf_model(ZERO_SHOT_MODEL_NAME, LOCAL_ZERO_SHOT_MODEL_DIR)
    
    emotion_clf = pipeline(
        "text-classification", 
        model=emotion_dir,
        tokenizer=emotion_dir,
        device=device,
        truncation=True,
        max_length=512
    )
    
    zero_shot_clf = pipeline(
        "zero-shot-classification", 
        model=zero_shot_dir,
        tokenizer=zero_shot_dir,
        device=device
    )
    
    return sia, emotion_clf, zero_shot_clf

@st.cache_resource(show_spinner="Downloading Hindi/Hinglish sentiment model on first use...")
def load_multilingual_sentiment_model():
    """Load a local-cached XLM-R model fine-tuned for multilingual sentiment."""
    try:
        import sentencepiece  # Required to read XLM-R's binary tokenizer file.
    except ImportError as exc:
        raise RuntimeError(
            "Hindi/Hinglish sentiment needs SentencePiece. Run: "
            "python -m pip install sentencepiece"
        ) from exc


    device = 0 if torch.cuda.is_available() else -1
    def build_pipeline():
        # Do not use AutoTokenizer for this model. Some Transformers releases
        # mistakenly pass XLM-R's SentencePiece binary to a tiktoken converter.
        tokenizer_class = _transformers.XLMRobertaTokenizer
        required_local_files = ("config.json", "sentencepiece.bpe.model")
        has_local_tokenizer = os.path.isdir(LOCAL_MULTILINGUAL_SENTIMENT_MODEL_DIR) and all(
            os.path.exists(os.path.join(LOCAL_MULTILINGUAL_SENTIMENT_MODEL_DIR, filename))
            for filename in required_local_files
        )
        has_local_weights = any(
            os.path.exists(os.path.join(LOCAL_MULTILINGUAL_SENTIMENT_MODEL_DIR, filename))
            for filename in ("pytorch_model.bin", "model.safetensors")
        )
        if has_local_tokenizer and has_local_weights:
            model_dir = LOCAL_MULTILINGUAL_SENTIMENT_MODEL_DIR
            tokenizer = tokenizer_class.from_pretrained(model_dir, use_fast=False)
        else:
            os.makedirs(LOCAL_MULTILINGUAL_SENTIMENT_MODEL_DIR, exist_ok=True)
            tokenizer = tokenizer_class.from_pretrained(
                MULTILINGUAL_SENTIMENT_MODEL_NAME,
                cache_dir=hf_cache_dir,
                use_fast=False,
            )
            model = _transformers.AutoModelForSequenceClassification.from_pretrained(
                MULTILINGUAL_SENTIMENT_MODEL_NAME, cache_dir=hf_cache_dir
            )
            tokenizer.save_pretrained(LOCAL_MULTILINGUAL_SENTIMENT_MODEL_DIR)
            model.save_pretrained(LOCAL_MULTILINGUAL_SENTIMENT_MODEL_DIR)
            model_dir = LOCAL_MULTILINGUAL_SENTIMENT_MODEL_DIR
        return pipeline(
            "text-classification",
            model=model_dir,
            tokenizer=tokenizer,
            device=device,
            truncation=True,
            max_length=512,
        )

    try:
        return build_pipeline()
    except Exception as exc:
        raise RuntimeError(
            "Hindi/Hinglish sentiment model could not be loaded. Its download may be "
            "incomplete, or this computer may be blocking Hugging Face downloads. "
            f"Underlying error: {exc}"
        ) from exc


@st.cache_resource(show_spinner="Downloading multilingual Hindi emotion model on first use...")
def load_multilingual_nli_model():
    """Load a smaller multilingual NLI model for Hindi/Hinglish zero-shot labels."""
    device = 0 if torch.cuda.is_available() else -1
    model_dir = ensure_local_hf_model(
        MULTILINGUAL_NLI_MODEL_NAME,
        LOCAL_MULTILINGUAL_NLI_MODEL_DIR,
        use_fast=False,
    )
    return pipeline(
        "zero-shot-classification",
        model=model_dir,
        tokenizer=model_dir,
        device=device,
    )

def normalize_multilingual_sentiment_label(label: str) -> str:
    """Normalize common pipeline labels to dashboard sentiment names."""
    normalized = label.lower().replace('_', '').replace('-', '').replace(' ', '')
    mapping = {
        'negative': 'Negative', 'label0': 'Negative',
        'neutral': 'Neutral', 'label1': 'Neutral',
        'positive': 'Positive', 'label2': 'Positive',
    }
    return mapping.get(normalized, label.title())


HINDI_EMOTION_LABELS = {
    "क्रोध": "Anger",
    "घृणा": "Disgust",
    "भय": "Fear",
    "खुशी": "Joy",
    "उदासी": "Sadness",
    "आश्चर्य": "Surprise",
    "तटस्थ": "Neutral",
}
HINDI_INTERPERSONAL_LABELS = [
    "सहानुभूतिपूर्ण", "सहायक", "सम्मानजनक",
    "उपेक्षापूर्ण", "तिरस्कारपूर्ण", "मज़ाक उड़ाने वाला",
]


def score_multilingual_inclusivity(classifier, text: str) -> dict:
    """Estimate interpersonal tone in Hindi/Hinglish with multilingual NLI."""
    result = classifier(
        text,
        HINDI_INTERPERSONAL_LABELS,
        multi_label=True,
        hypothesis_template="यह संवाद {} है।",
    )
    scores = dict(zip(result["labels"], result["scores"]))
    positive = np.mean([scores[label] for label in HINDI_INTERPERSONAL_LABELS[:3]])
    negative = np.mean([scores[label] for label in HINDI_INTERPERSONAL_LABELS[3:]])
    return {
        "Empathy_Score": round(float(positive - negative), 3),
        "Dominant_Trait": result["labels"][0],
    }

# English-only models are loaded only when an English script reaches that
# analysis path. Hindi OCR uploads use multilingual models instead.
sia = emotion_classifier = zs_classifier = None

# Constants
POSITIVE_LABELS = ["empathetic", "supportive", "respectful"]
NEGATIVE_LABELS = ["condescending", "dismissive", "mocking"]
ALL_LABELS = POSITIVE_LABELS + NEGATIVE_LABELS

# ==========================================
# 2. CORE BACKEND FUNCTIONS
# ==========================================
def is_page_number_or_copyright(line: str) -> bool:
    if re.match(r'^(?:\d{1,3}\.?|\[?\d{1,3}\]?|\(\d{1,3}\))$', line):
        return True
    if re.search(r'(copyright|©|all rights reserved)', line, re.IGNORECASE):
        return True
    return False


def is_non_dialogue_ocr_line(line: str) -> bool:
    """Exclude common OCR/front-matter noise before screenplay parsing.

    OCR text from books and legacy-font PDFs includes ISBNs, prices, publisher
    lines, and page numbers.  Those are not dialogue and must never inherit the
    preceding character heading.
    """
    compact = re.sub(r'\s+', '', line)
    if not compact or is_page_number_or_copyright(line):
        return True
    if re.fullmatch(r'[\d\s.,:/-]+', line):
        return True
    if re.search(r'\b(?:isbn|copyright|all rights reserved|printed|published|edition|price|rs\.?|rupees)\b', line, re.IGNORECASE):
        return True
    if re.search(r'(?:आईएसबीएन|कॉपीराइट|सर्वाधिकार|प्रकाशक|प्रकाशन|संस्करण|मूल्य)', line):
        return True
    # ISBNs, catalogue numbers, and prices are mostly digits/punctuation even
    # when OCR inserts spaces between them.
    alpha_or_devanagari = len(re.findall(r'[A-Za-z\u0900-\u097f]', compact))
    if len(compact) >= 6 and alpha_or_devanagari / len(compact) < 0.35:
        return True
    return False

def clean_character_name(line: str) -> str:
    cleaned = re.sub(r'\(.*?\)', '', line).strip()
    return re.sub(r'[^A-Z0-9\s\-/]', '', cleaned).strip()

def clean_dialogue_text(text: str) -> str:
    text = re.sub(r'\(.*?\)', '', text).lower()
    # Preserve pauses: the old cleaning rule removed dashes before hesitation
    # analysis could see them.
    text = text.replace('…', '...').replace('—', '--').replace('–', '--')
    # Keep Devanagari text so Hindi screenplays are not erased during parsing.
    text = re.sub(r'[^a-z0-9\u0900-\u097f\s\.,!\?\'\"-]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def parse_screenplay(file_path: str) -> pd.DataFrame:
    movie_name = os.path.basename(file_path).split('.')[0]
    extracted_data = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    if is_kruti_dev(content):
        content = kruti_to_unicode(content)

    raw_lines = content.splitlines()
    cleaned_lines = []
    for line in raw_lines:
        stripped = line.strip()
        cleaned_lines.append((line, stripped))

    # Check if the file is in QUOTED_CSV format ("number" "CHARACTER" "dialogue")
    quoted_csv_score = 0
    total_lines_checked = 0
    for orig, stripped in cleaned_lines[:15]:
        if not stripped: continue
        if re.match(r'^\s*"\d+"\s+"[^"]+"\s+".*"', stripped):
            quoted_csv_score += 1
        total_lines_checked += 1
    
    is_quoted_csv = (total_lines_checked > 0 and quoted_csv_score / total_lines_checked >= 0.5)

    if is_quoted_csv:
        current_scene = "UNKNOWN_SCENE"
        for orig, stripped in cleaned_lines:
            match = re.match(r'^\s*"\d+"\s+"([^"]+)"\s+"(.*)"\s*$', stripped)
            if match:
                char_name = match.group(1).strip()
                dialogue_text = match.group(2).strip()
                extracted_data.append({
                    'Movie': movie_name,
                    'Scene': current_scene,
                    'Character': char_name,
                    'Dialogue': dialogue_text
                })
    else:
        filtered_lines = []
        for orig, stripped in cleaned_lines:
            if not stripped:
                filtered_lines.append(("", ""))
                continue
            if is_non_dialogue_ocr_line(stripped):
                # A metadata line also ends any preceding dialogue block.
                # Without this separator, it is incorrectly appended to the
                # prior speaker's dialogue.
                filtered_lines.append(("", ""))
                continue
            filtered_lines.append((orig, stripped))

        # Auto-detect if it's SAME_LINE_ALL_CAPS format (like simple script logs)
        blocks = []
        current_block = []
        for orig, stripped in filtered_lines:
            if not stripped:
                if current_block:
                    blocks.append(current_block)
                    current_block = []
            else:
                current_block.append((orig, stripped))
        if current_block:
            blocks.append(current_block)

        same_line_caps_score = 0
        total_blocks_checked = 0
        for block in blocks[:10]:
            if not block: continue
            first_line_orig, first_line_stripped = block[0]
            match = re.match(r'^([A-Z]{2,15})\b\s+(.+)$', first_line_stripped)
            if match:
                same_line_caps_score += 1
            total_blocks_checked += 1

        is_same_line_all_caps = (total_blocks_checked > 0 and same_line_caps_score / total_blocks_checked >= 0.7)

        if is_same_line_all_caps:
            current_scene = "UNKNOWN_SCENE"
            for block in blocks:
                if not block: continue
                first_line_orig, first_line_stripped = block[0]
                
                if re.match(r'^(?:INT\.|EXT\.|INT/EXT\.|INT |EXT |I/E ).*', first_line_stripped, re.IGNORECASE):
                    current_scene = first_line_stripped
                    continue
                    
                match = re.match(r'^([A-Z]{2,15})\b\s*(.*)$', first_line_stripped)
                if match:
                    char_name = match.group(1)
                    dialogue_parts = [match.group(2)]
                    for orig, stripped in block[1:]:
                        dialogue_parts.append(stripped)
                    dialogue_text = " ".join([p for p in dialogue_parts if p])
                    if dialogue_text.strip():
                        extracted_data.append({
                            'Movie': movie_name,
                            'Scene': current_scene,
                            'Character': char_name,
                            'Dialogue': dialogue_text
                        })
        else:
            # Standard screenplay parser (handles both mixed-case and standard all-caps screenplays)
            current_scene = "UNKNOWN_SCENE"
            current_character = None
            current_dialogue_block = []

            for orig, stripped in filtered_lines:
                if not stripped:
                    if current_character and current_dialogue_block:
                        extracted_data.append({
                            'Movie': movie_name,
                            'Scene': current_scene,
                            'Character': current_character,
                            'Dialogue': " ".join(current_dialogue_block)
                        })
                        current_dialogue_block = []
                    current_character = None
                    continue

                # Scene Headings
                if re.match(r'^(?:INT\.|EXT\.|INT/EXT\.|INT |EXT |I/E ).*', stripped, re.IGNORECASE):
                    current_scene = stripped
                    current_character = None
                    continue

                # Transitions
                if re.match(r'.*(?:CUT TO:|FADE IN:|FADE OUT\.|DISSOLVE TO:)$', stripped, re.IGNORECASE):
                    current_character = None
                    continue

                # Detect Character Names
                is_upper = stripped.isupper() and not re.search(r'[a-z]', stripped)
                words = stripped.split()
                has_sentence_punct = any(char in stripped for char in ['?', '!', ',']) or (stripped.endswith('.') and not any(stripped.endswith(abbr) for abbr in ['V.O.', 'O.S.', 'CONT\'D.', 'CONTD.']))
                is_char_name = is_upper and len(words) <= 4 and not has_sentence_punct

                # Hindi plays often write a speaker as "पात्र : संवाद" rather
                # than using all-capital screenplay headings.
                hindi_speaker_match = re.match(r'^([\u0900-\u097fA-Za-z0-9०-९\s\/\-]{1,38})\s*[:：%]\s*(.*)$', stripped)
                if hindi_speaker_match:
                    flush_dialogue = current_character and current_dialogue_block
                    if flush_dialogue:
                        extracted_data.append({
                            'Movie': movie_name,
                            'Scene': current_scene,
                            'Character': current_character,
                            'Dialogue': " ".join(current_dialogue_block)
                        })
                        current_dialogue_block = []
                    current_character = hindi_speaker_match.group(1).strip()
                    inline_dialogue = hindi_speaker_match.group(2).strip()
                    if inline_dialogue:
                        current_dialogue_block.append(inline_dialogue)
                    continue
                
                if is_char_name:
                    cleaned_char = clean_character_name(stripped)
                    if len(cleaned_char) > 1:
                        current_character = cleaned_char
                    continue

                if current_character:
                    current_dialogue_block.append(stripped)

            # Catch trailing dialogue
            if current_character and current_dialogue_block:
                extracted_data.append({
                    'Movie': movie_name,
                    'Scene': current_scene,
                    'Character': current_character,
                    'Dialogue': " ".join(current_dialogue_block)
                })

    df = pd.DataFrame(extracted_data)
    if not df.empty:
        df['Dialogue'] = df['Dialogue'].apply(clean_dialogue_text)
        df = df[df['Dialogue'].str.strip() != ''].reset_index(drop=True)
    return df

def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract selectable text from a screenplay PDF without changing the parser."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support needs pypdf. Run: python -m pip install pypdf") from exc

    try:
        reader = PdfReader(BytesIO(file_bytes))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as exc:
        raise RuntimeError(f"Could not read this PDF: {exc}") from exc
    if not text:
        raise RuntimeError("No selectable text was found in this PDF. It may be a scanned image PDF; use a text-based PDF or OCR it first.")
    return text

def is_garbled_text(text: str) -> bool:
    """Identify legacy-font PDF text that is unsuitable for language analysis."""
    compact = re.sub(r'\s+', '', text)
    if len(compact) < 40:
        return True
    # Real Hindi PDFs contain Devanagari. Legacy KrutiDev-style maps commonly
    # extract as improbable Latin strings.  A vowel count alone is not enough:
    # encoded Hindi may contain plenty of a/e/i/o/u, so also require evidence
    # of actual English words before treating a Latin-only PDF as readable.
    devanagari_share = len(re.findall(r'[\u0900-\u097f]', text)) / max(len(compact), 1)
    if devanagari_share > 0:
        return False
    words = re.findall(r"[A-Za-z']+", text.lower())
    english_markers = {
        'the', 'and', 'that', 'this', 'with', 'from', 'you', 'your', 'have',
        'are', 'was', 'for', 'not', 'but', 'all', 'they', 'she', 'him', 'her',
        'script', 'scene', 'page', 'said', 'said', 'what', 'when', 'where',
    }
    english_share = sum(word in english_markers for word in words) / max(len(words), 1)
    legacy_markers = sum("'" in word or any(letter in word for letter in 'qwx') for word in words) / max(len(words), 1)
    return len(words) >= 20 and english_share < 0.015 and legacy_markers > 0.08

def script_category(score: float) -> str:
    """Return a non-numeric, dialogue-signal category for the dashboard."""
    if score < 35:
        return "Developing dialogue signals"
    if score < 50:
        return "Emerging dialogue signals"
    if score < 65:
        return "Balanced dialogue signals"
    if score < 80:
        return "Strong dialogue signals"
    return "Highly engaging dialogue signals"

def ocr_hindi_pdf(file_bytes: bytes, status_callback=None) -> str:
    """OCR a legacy Hindi PDF using installed Ghostscript and Tesseract.

    This deliberately uses command-line tools already present on the user's
    machine, avoiding a pip download for large OCR dependencies.
    """
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ghostscript_path = r"C:\Program Files\gs\gs10.08.0\bin\gswin64c.exe"
    if not os.path.isfile(tesseract_path) or not os.path.isfile(ghostscript_path):
        raise RuntimeError("This PDF needs Hindi OCR. Install Ghostscript and Tesseract OCR with Hindi language data, then restart the app.")

    with tempfile.TemporaryDirectory(prefix="hindi_pdf_ocr_") as temp_dir:
        source_path = Path(temp_dir) / "source.pdf"
        source_path.write_bytes(file_bytes)
        try:
            from pypdf import PdfReader
            total_pages = len(PdfReader(BytesIO(file_bytes)).pages)
        except Exception as exc:
            raise RuntimeError("Could not determine the number of pages for Hindi OCR.") from exc
        pages = []
        for page_number in range(1, total_pages + 1):
            if status_callback:
                status_callback(page_number, total_pages)
            image_path = Path(temp_dir) / f"page_{page_number:03d}.png"
            render = subprocess.run([
                ghostscript_path, "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE",
                "-sDEVICE=png16m", "-r180", f"-dFirstPage={page_number}",
                f"-dLastPage={page_number}", f"-sOutputFile={image_path}", str(source_path),
            ], capture_output=True, text=True, check=False)
            if render.returncode != 0 or not image_path.is_file():
                raise RuntimeError(f"Could not render PDF page {page_number} for Hindi OCR.")
            ocr = subprocess.run(
                [tesseract_path, str(image_path), "stdout", "-l", "hin+eng", "--psm", "6"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
            )
            if ocr.returncode != 0:
                raise RuntimeError(f"Hindi OCR failed on page {page_number}.")
            pages.append(ocr.stdout)
    return "\n\n".join(pages).strip()

def parse_unattributed_pdf_dialogues(content: str, movie_name: str) -> pd.DataFrame:
    """Fallback for reading-copy PDFs that omit speaker names.

    Such PDFs can support line-level and whole-script analysis, but not reliable
    character-level conclusions.  Every row is therefore labelled UNATTRIBUTED.
    """
    rows, quote_buffer = [], []

    def add_dialogue(parts: list[str]) -> None:
        dialogue = clean_dialogue_text(" ".join(parts).strip(' "'))
        if not is_non_dialogue_ocr_line(dialogue) and len(dialogue.split()) >= 2:
            rows.append({
                'Movie': movie_name,
                'Scene': 'UNAVAILABLE_IN_SOURCE_PDF',
                'Character': 'UNATTRIBUTED',
                'Dialogue': dialogue,
            })

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or is_non_dialogue_ocr_line(line) or re.search(r'https?://|scripts\.com|^page\s+\d+\s*/\s*\d+|^reading copy$|^a script made readable$', line, re.IGNORECASE):
            continue
        if quote_buffer:
            quote_buffer.append(line)
            if line.count('"') % 2 == 1:
                add_dialogue(quote_buffer)
                quote_buffer = []
            continue
        if line.count('"') % 2 == 1:
            quote_buffer = [line]
        else:
            add_dialogue([line])
    if quote_buffer:
        add_dialogue(quote_buffer)
    return pd.DataFrame(rows, columns=['Movie', 'Scene', 'Character', 'Dialogue'])

def ollama_is_available() -> bool:
    """Return whether the local Ollama server is running; never contacts the internet."""
    try:
        with urlrequest.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.5):
            return True
    except (urlerror.URLError, TimeoutError):
        return False


def extract_ollama_dialogues(raw_response: str) -> list:
    """Decode an Ollama JSON response, salvaging complete records if it was cut off."""
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        parsed = json.loads(cleaned)
        return parsed.get("dialogues", []) if isinstance(parsed, dict) else []
    except json.JSONDecodeError:
        # Small local models can stop while writing a long JSON array. Recover
        # every complete object rather than failing the whole uploaded PDF.
        match = re.search(r'"dialogues"\s*:\s*\[', cleaned)
        if not match:
            return []
        decoder = json.JSONDecoder()
        records, position = [], match.end()
        while position < len(cleaned):
            while position < len(cleaned) and cleaned[position] in " \t\r\n,":
                position += 1
            if position >= len(cleaned) or cleaned[position] == "]":
                break
            try:
                item, position = decoder.raw_decode(cleaned, position)
            except json.JSONDecodeError:
                break
            if isinstance(item, dict):
                records.append(item)
        return records


def representative_ocr_excerpt(content: str, max_characters: int = 9000) -> str:
    """Keep OCR context small, while covering more than a PDF's front matter."""
    if len(content) <= max_characters:
        return content
    chunk_size = max_characters // 4
    starts = [
        0,
        max(0, len(content) // 3 - chunk_size // 2),
        max(0, (2 * len(content)) // 3 - chunk_size // 2),
        max(0, len(content) - chunk_size),
    ]
    excerpts = [content[start:start + chunk_size].strip() for start in starts]
    return "\n\n--- OCR EXCERPT: NEXT PART OF THE PDF ---\n\n".join(excerpt for excerpt in excerpts if excerpt)


def parse_with_ollama(content: str, movie_name: str, model_name: str) -> pd.DataFrame:
    """Use a local LLM to recover screenplay structure from OCR text."""
    # Gemma 1B is installed locally. A bounded source/output keeps it from
    # truncating an oversized JSON array halfway through a response.
    source = representative_ocr_excerpt(content)
    prompt = f'''You extract screenplay dialogue from OCR text. Return JSON only:
{{"dialogues":[{{"character":"speaker name or UNATTRIBUTED","scene":"act or scene name, or UNKNOWN_SCENE","dialogue":"spoken dialogue"}}]}}

Keep only spoken dialogue. Exclude ISBN, prices, page numbers, copyright,
publishing information, titles, and tables of contents. Infer a speaker only
when the source clearly identifies one; otherwise use UNATTRIBUTED. Preserve
Hindi, Hinglish, and English wording. Do not translate or invent dialogue.
The OCR text contains excerpts from different parts of one PDF. Return at most
25 dialogue records from any excerpt. Ensure the JSON is complete and valid.

OCR text:
{source}'''
    payload = json.dumps({
        "model": model_name, "prompt": prompt, "format": "json", "stream": False,
        "options": {"temperature": 0, "num_predict": 4096},
    }).encode("utf-8")
    request = urlrequest.Request(
        "http://127.0.0.1:11434/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlrequest.urlopen(request, timeout=300) as response:
            response_body = json.loads(response.read().decode("utf-8"))
            items = extract_ollama_dialogues(str(response_body["response"]))
    except (urlerror.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama could not return valid structured dialogue: {exc}") from exc

    if not items:
        raise RuntimeError("Ollama returned no complete dialogue records. Try the normal parser or use a clearer PDF.")

    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        dialogue = clean_dialogue_text(str(item.get("dialogue", "")))
        character = str(item.get("character", "UNATTRIBUTED")).strip() or "UNATTRIBUTED"
        scene = str(item.get("scene", "UNKNOWN_SCENE")).strip() or "UNKNOWN_SCENE"
        if len(dialogue.split()) >= 2 and not is_non_dialogue_ocr_line(dialogue):
            rows.append({"Movie": movie_name, "Scene": scene, "Character": character, "Dialogue": dialogue})
    return pd.DataFrame(rows, columns=["Movie", "Scene", "Character", "Dialogue"])


def parse_with_gemini(content: str, movie_name: str) -> pd.DataFrame:
    """Use Gemini only when the user supplies an API key through the environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except FileNotFoundError:
            api_key = None
    if not api_key:
        raise RuntimeError("Gemini needs GEMINI_API_KEY. Put it in .streamlit/secrets.toml or start the app with start_gemini_app.ps1.")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Gemini support needs google-genai. Run start_gemini_app.ps1 once.") from exc

    prompt = f'''Extract screenplay dialogue from this OCR text. Return JSON only:
{{"dialogues":[{{"character":"speaker name or UNATTRIBUTED","scene":"act or scene name, or UNKNOWN_SCENE","dialogue":"spoken dialogue"}}]}}
Keep only spoken dialogue. Exclude ISBN, prices, page numbers, copyright,
publishing information, titles, and tables of contents. Infer a speaker only
when clearly identified. Preserve Hindi, Hinglish, and English wording. Do not
translate or invent dialogue.

OCR text:
{content[:30000]}'''
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
        )
        parsed = json.loads(response.text)
    except Exception as exc:
        raise RuntimeError(f"Gemini could not return valid structured dialogue: {exc}") from exc

    rows = []
    for item in parsed.get("dialogues", []):
        if not isinstance(item, dict):
            continue
        dialogue = clean_dialogue_text(str(item.get("dialogue", "")))
        character = str(item.get("character", "UNATTRIBUTED")).strip() or "UNATTRIBUTED"
        scene = str(item.get("scene", "UNKNOWN_SCENE")).strip() or "UNKNOWN_SCENE"
        if len(dialogue.split()) >= 2 and not is_non_dialogue_ocr_line(dialogue):
            rows.append({"Movie": movie_name, "Scene": scene, "Character": character, "Dialogue": dialogue})
    return pd.DataFrame(rows, columns=["Movie", "Scene", "Character", "Dialogue"])


def extract_lexical_features(text: str) -> dict:
    words, sentences = word_tokenize(str(text)), sent_tokenize(str(text))
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    unique_words = set(w.lower() for w in words if w.isalnum())
    
    hesitation_matches = re.findall(r'\.\.\.|--|\b(?:um+|uh+|erm+|hmm+)\b', str(text).lower())
    return {
        'Word_Count': word_count,
        'Lexical_Diversity_TTR': len(unique_words) / word_count if word_count > 0 else 0,
        'Hesitation_Markers': len(hesitation_matches),
        'Hesitation_Rate_Per_100_Words': (len(hesitation_matches) / word_count * 100) if word_count > 0 else 0,
        'Is_Question': 1 if '?' in str(text) else 0
    }

def dialogue_signal_score(df: pd.DataFrame) -> tuple[float, dict]:
    """A transparent descriptive score, deliberately not an IMDb prediction."""
    total_words = max(int(df['Word_Count'].sum()), 1)
    character_share = df['Character'].value_counts(normalize=True).iloc[0]
    components = {
        'Lexical variety': round(df['Lexical_Diversity_TTR'].mean() * 100, 1),
        'Character participation balance': round((1 - character_share) * 100, 1),
        'Question / turn-taking signal': round(df['Is_Question'].mean() * 100, 1),
        'Hesitation markers per 100 words': round(df['Hesitation_Markers'].sum() / total_words * 100, 2),
    }
    # The first three signals describe dialogue variety and distribution. This is
    # a UI summary only; it must never be presented as an IMDb score.
    score = round(0.5 * components['Lexical variety'] + 0.35 * components['Character participation balance'] + 0.15 * components['Question / turn-taking signal'], 1)
    return score, components

def evenly_spaced_indices(length: int, maximum: int) -> list[int]:
    """Choose representative script positions without making CPU inference unbounded."""
    if length <= maximum:
        return list(range(length))
    return np.linspace(0, length - 1, maximum, dtype=int).tolist()

def contains_devanagari(text: str) -> bool:
    return bool(re.search(r'[\u0900-\u097f]', text))

def score_inclusivity(text: str) -> dict:
    try:
        result = zs_classifier(text, candidate_labels=ALL_LABELS, multi_label=True)
        scores = dict(zip(result['labels'], result['scores']))
        
        pos = sum(scores[l] for l in POSITIVE_LABELS)
        neg = sum(scores[l] for l in NEGATIVE_LABELS)
        
        scores['Empathy_Score'] = pos - neg
        scores['Dominant_Trait'] = max(scores, key=lambda k: scores[k] if k in ALL_LABELS else -1)
        return scores
    except:
        return {l: 0.0 for l in ALL_LABELS} | {'Empathy_Score': 0.0, 'Dominant_Trait': 'None'}

# ==========================================
# 3. FRONTEND UI & LOGIC
# ==========================================
st.markdown("""
<section class="hero">
  <h1>🎬 Media Empathy Analyzer</h1>
  <p>Explore emotion, communication patterns, hesitation signals, and interpersonal tone across every screenplay dialogue.</p>
  <span>Screenplay intelligence dashboard</span>
</section>
""", unsafe_allow_html=True)
st.markdown('<div class="section-note">Upload a <b>.txt screenplay</b>, then run the analysis. Results are organised by character, scene, and dialogue line.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Analysis status")
    st.caption("Everything is set automatically. Upload your screenplay and start the analysis.")
    st.markdown("""
    <div class="auto-card">
      <b>Auto mode enabled</b><br>
      Longer dialogue lines are selected automatically for detailed interpersonal-tone analysis.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Includes emotion, sentiment, character communication, hesitation patterns, and empathy signals.")
    st.markdown("---")
    use_ollama_parser = st.checkbox(
        "Use local Ollama for PDF structure",
        value=False,
        help="Uses a local LLM to identify acts, scenes, speakers, and dialogue in difficult OCR PDFs.",
    )
    ollama_model_name = st.text_input("Ollama model", value="gemma3:1b", disabled=not use_ollama_parser)
    if use_ollama_parser:
        if ollama_is_available():
            st.success("Ollama is running locally.")
        else:
            st.warning("Ollama is not running yet. Start Ollama, then retry with the installed `gemma3:1b` model.")
    use_gemini_parser = st.checkbox(
        "Use Gemini for PDF structure",
        value=False,
        help="Uses Gemini API for speaker, act, scene, and dialogue extraction. Requires GEMINI_API_KEY and internet.",
    )
    if use_gemini_parser:
        st.caption("Your key is read only from GEMINI_API_KEY; it is not stored in this app or repository.")

uploaded_file = st.file_uploader("Upload screenplay", type=["txt", "pdf"], help="Text files and selectable-text PDF screenplays are supported.")

if uploaded_file is not None:
    if st.button("Run Full NLP Pipeline", type="primary"):
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            # --- STEP 1: Parsing ---
            status.text("Step 1/4: Parsing screenplay structure...")
            uploaded_bytes = uploaded_file.getvalue()
            suffix = os.path.splitext(uploaded_file.name)[1].lower()
            pdf_text = None
            if suffix == ".pdf":
                pdf_text = extract_pdf_text(uploaded_bytes)
                if is_kruti_dev(pdf_text):
                    status.text("Kruti Dev legacy Hindi font detected. Auto-converting to Unicode Devanagari in 0.1s...")
                    pdf_text = kruti_to_unicode(pdf_text)
                elif is_garbled_text(pdf_text):
                    status.text("Legacy font / scanned PDF detected. Running Hindi OCR fallback...")
                    def update_ocr_status(current_page: int, total_pages: int) -> None:
                        progress_bar.progress(int(current_page / total_pages * 20))
                        status.text(f"Hindi OCR in progress: page {current_page}/{total_pages}. Please keep this tab open.")
                    pdf_text = ocr_hindi_pdf(uploaded_bytes, update_ocr_status)
                    if not pdf_text:
                        raise RuntimeError("Hindi OCR could not recover readable text from this PDF.")
                content = pdf_text.encode("utf-8")
            else:
                raw_text = uploaded_bytes.decode("utf-8", errors="ignore")
                if is_kruti_dev(raw_text):
                    status.text("Kruti Dev text detected. Auto-converting to Unicode Devanagari...")
                    raw_text = kruti_to_unicode(raw_text)
                    content = raw_text.encode("utf-8")
                else:
                    content = uploaded_bytes

            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
                
            df = parse_screenplay(tmp_path)
            os.remove(tmp_path)
            df['Movie'] = os.path.splitext(uploaded_file.name)[0]
            unattributed_pdf = False
            if use_ollama_parser and use_gemini_parser:
                raise RuntimeError("Choose either Ollama or Gemini for AI-enhanced parsing, not both.")
            if use_ollama_parser:
                if not ollama_is_available():
                    raise RuntimeError("Ollama is not running. Start Ollama, download the selected local model, then try again.")
                status.text("Step 1/4: Local Ollama is identifying acts, speakers, and dialogue from the PDF...")
                try:
                    llm_df = parse_with_ollama(
                        pdf_text if pdf_text is not None else content.decode("utf-8", errors="ignore"),
                        os.path.splitext(uploaded_file.name)[0],
                        ollama_model_name,
                    )
                except RuntimeError as exc:
                    st.warning(f"Ollama response could not be used ({exc}). Continuing with local Hindi OCR parsing instead.")
                    llm_df = pd.DataFrame(columns=["Movie", "Scene", "Character", "Dialogue"])
                # One or two tiny model records are not a successful parse of
                # a long document. They produce misleading dashboard metrics
                # (for example, "1 dialogue line" for a 96-page Hindi PDF).
                minimum_llm_rows = 5 if pdf_text is not None and len(pdf_text) > 5000 else 1
                if len(llm_df) < minimum_llm_rows:
                    st.warning("Ollama did not recover enough speaker-labelled dialogue. Continuing with local Hindi OCR parsing instead.")
                    if pdf_text is not None:
                        df = parse_unattributed_pdf_dialogues(pdf_text, os.path.splitext(uploaded_file.name)[0])
                        unattributed_pdf = not df.empty
                else:
                    df = llm_df
                    st.info("Local Ollama parser used. Speaker and act/scene labels are AI-extracted and should be reviewed for OCR errors.")
            elif use_gemini_parser:
                status.text("Step 1/4: Gemini is identifying acts, speakers, and dialogue from the PDF...")
                llm_df = parse_with_gemini(
                    pdf_text if pdf_text is not None else content.decode("utf-8", errors="ignore"),
                    os.path.splitext(uploaded_file.name)[0],
                )
                if llm_df.empty:
                    raise RuntimeError("Gemini did not find usable dialogue. Try the normal parser or provide a PDF with clearer speaker labels.")
                df = llm_df
                st.info("Gemini parser used. Speaker and act/scene labels are AI-extracted and should be reviewed for OCR errors.")
            elif pdf_text is not None and len(df) <= 1:
                df = parse_unattributed_pdf_dialogues(pdf_text, os.path.splitext(uploaded_file.name)[0])
                unattributed_pdf = not df.empty
            progress_bar.progress(25)
            
            if df.empty:
                st.error("No dialogue could be extracted. Please check the file formatting.")
                st.stop()

            if unattributed_pdf:
                st.warning("This PDF does not include character names, so it is being analysed as unattributed dialogue lines. Whole-script emotion and language results are valid, but character comparisons and the AI script rating are unavailable for this source format.")
                
            # --- STEP 2: Sentiment & Emotion ---
            # Transformer models are expensive on laptop CPUs.  Analyse a
            # representative sample and make the limit visible rather than
            # appearing stuck for a long screenplay.
            is_hindi_script = contains_devanagari(" ".join(df['Dialogue'].head(100).tolist()))
            if is_hindi_script:
                max_hindi_lines = 120 if len(df) > 120 else len(df)
                hindi_indices = evenly_spaced_indices(len(df), max_hindi_lines)
                status.text(f"Step 2/4: Hindi / Hinglish detected — downloading or loading multilingual sentiment model, then analysing {len(hindi_indices)} representative lines...")
                multilingual_sentiment_classifier = load_multilingual_sentiment_model()
                df['Vader'] = 0.0
                df['Sentiment'] = 'Not sampled (CPU limit)'
                hindi_results = []
                batch_size = 6
                for start in range(0, len(hindi_indices), batch_size):
                    batch_indices = hindi_indices[start:start + batch_size]
                    hindi_results.extend(multilingual_sentiment_classifier(
                        df.loc[batch_indices, 'Dialogue'].tolist(), batch_size=batch_size
                    ))
                    completed = min(start + batch_size, len(hindi_indices))
                    progress_bar.progress(25 + int(completed / max(len(hindi_indices), 1) * 25))
                    status.text(f"Step 2/4: Hindi / Hinglish sentiment analysis {completed}/{len(hindi_indices)} lines...")
                df.loc[hindi_indices, 'Sentiment'] = [
                    normalize_multilingual_sentiment_label(result['label']) for result in hindi_results
                ]
                df['Emotion'] = 'Not sampled (CPU limit)'
                st.info("Hindi / Devanagari detected: multilingual sentiment is enabled. Hindi emotion and interpersonal tone are calculated for a small representative sample in Step 4, because they are slower on a CPU.")
            else:
                sia, emotion_classifier, zs_classifier = load_models()
                using_gpu = torch.cuda.is_available()
                max_emotion_lines = 300 if using_gpu else 60
                emotion_indices = evenly_spaced_indices(len(df), max_emotion_lines)
                status.text(f"Step 2/4: Analysing sentiment for all {len(df):,} lines and emotion for a representative {len(emotion_indices):,}-line sample...")
                df['Vader'] = df['Dialogue'].apply(lambda x: sia.polarity_scores(x)['compound'])
                df['Sentiment'] = pd.cut(df['Vader'], bins=[-1, -0.05, 0.05, 1], labels=['Negative', 'Neutral', 'Positive'])
                df['Emotion'] = 'Not sampled'
                emotion_results = []
                batch_size = 24 if using_gpu else 6
                for start in range(0, len(emotion_indices), batch_size):
                    batch_indices = emotion_indices[start:start + batch_size]
                    batch_text = df.loc[batch_indices, 'Dialogue'].tolist()
                    emotion_results.extend(emotion_classifier(batch_text, batch_size=batch_size))
                    completed = min(start + batch_size, len(emotion_indices))
                    progress_bar.progress(25 + int(completed / max(len(emotion_indices), 1) * 25))
                    status.text(f"Step 2/4: Emotion analysis {completed}/{len(emotion_indices)} representative lines...")
                df.loc[emotion_indices, 'Emotion'] = [result['label'].capitalize() for result in emotion_results]
            progress_bar.progress(50)
            
            # --- STEP 3: Lexical Features ---
            status.text("Step 3/4: Calculating Lexical & Communication Features...")
            features_df = pd.DataFrame(df['Dialogue'].apply(extract_lexical_features).tolist())
            df = pd.concat([df, features_df], axis=1)
            # A chronological split works even when a PDF has no scene headings.
            df['Script_Progress_Percent'] = ((np.arange(len(df)) + 1) / len(df) * 100).round(1)
            df['Script_Section'] = pd.cut(
                df['Script_Progress_Percent'],
                bins=[0, 33.34, 66.67, 100],
                labels=['Beginning (0-33%)', 'Middle (34-66%)', 'Ending (67-100%)'],
                include_lowest=True,
            )
            progress_bar.progress(75)
            
            # --- STEP 4: Zero-Shot Inclusivity Scoring ---
            eligible_zs = df[df['Word_Count'] >= 5].copy()
            # BART-large zero-shot inference is substantially slower than the
            # emotion classifier. Keep CPU analysis responsive while retaining
            # wider coverage when a CUDA GPU is available.
            max_interpersonal_lines = 75 if torch.cuda.is_available() else 8
            auto_sample_size = min(max_interpersonal_lines, len(eligible_zs))
            if is_hindi_script:
                max_hindi_nli_lines = 8 if not torch.cuda.is_available() else 40
                hindi_nli_size = min(max_hindi_nli_lines, len(eligible_zs))
                status.text(f"Step 4/4: Loading multilingual Hindi emotion and empathy model, then analysing {hindi_nli_size} representative lines...")
                if hindi_nli_size:
                    multilingual_nli = load_multilingual_nli_model()
                    nli_positions = evenly_spaced_indices(len(eligible_zs), hindi_nli_size)
                    df_zs = eligible_zs.iloc[nli_positions].copy()
                    nli_rows = []
                    for count, (row_index, dialogue) in enumerate(zip(df_zs.index, df_zs['Dialogue']), start=1):
                        emotion_result = multilingual_nli(
                            dialogue,
                            list(HINDI_EMOTION_LABELS),
                            hypothesis_template="यह संवाद {} दर्शाता है।",
                        )
                        emotion = HINDI_EMOTION_LABELS[emotion_result['labels'][0]]
                        df_zs.loc[row_index, 'Emotion'] = emotion
                        df.loc[row_index, 'Emotion'] = emotion
                        nli_rows.append(score_multilingual_inclusivity(multilingual_nli, dialogue))
                        progress_bar.progress(75 + int(count / hindi_nli_size * 25))
                        status.text(f"Step 4/4: Hindi emotion and interpersonal-tone analysis {count}/{hindi_nli_size} lines...")
                    df_zs = pd.concat([df_zs, pd.DataFrame(nli_rows, index=df_zs.index)], axis=1).reset_index(drop=True)
                else:
                    df_zs = pd.DataFrame()
            else:
                status.text(f"Step 4/4: Analysing {auto_sample_size} representative substantial dialogue lines for interpersonal tone...")
                if auto_sample_size:
                    zs_positions = evenly_spaced_indices(len(eligible_zs), auto_sample_size)
                    df_zs = eligible_zs.iloc[zs_positions].copy()
                    zs_results = []
                    for count, text in enumerate(df_zs['Dialogue'], start=1):
                        zs_results.append(score_inclusivity(text))
                        progress_bar.progress(75 + int(count / auto_sample_size * 25))
                        status.text(f"Step 4/4: Interpersonal-tone analysis {count}/{auto_sample_size} lines...")
                    df_zs = pd.concat([df_zs.reset_index(drop=True), pd.DataFrame(zs_results)], axis=1)
                else:
                    df_zs = pd.DataFrame()
            
            progress_bar.progress(100)
            status.success("Analysis Complete!")
            
            # ==========================================
            # 4. DASHBOARD VISUALIZATIONS
            # ==========================================
            st.markdown("---")
            signal_score, signal_components = dialogue_signal_score(df)
            tab0, tab1, tab2, tab3, tab4 = st.tabs([
                "Script Overview", "Emotional Landscape", "Communication Profiles",
                "Inclusivity & Empathy", "Dialogue Explorer"
            ])
            sns.set_theme(style="whitegrid")
            
            top_chars = df['Character'].value_counts().head(8).index
            top_df = df[df['Character'].isin(top_chars)]
            
            with tab0:
                st.subheader("Screenplay Certification & Parents Guide")
                parents_guide = compute_parents_guide(df)
                cbfc_code = parents_guide['cbfc_code']
                mpaa_code = parents_guide['mpaa_code']
                advisory_summary = parents_guide['summary']

                cbfc_bg = "#d32f2f" if cbfc_code == "A" else ("#f57c00" if "16" in cbfc_code else ("#ffa000" if "13" in cbfc_code else ("#fbc02d" if "7" in cbfc_code else "#2e7d32")))
                mpaa_bg = "#d32f2f" if mpaa_code in ("R", "NC-17") else ("#f57c00" if mpaa_code == "PG-13" else ("#fbc02d" if mpaa_code == "PG" else "#2e7d32"))

                cert_col1, cert_col2 = st.columns([1.1, 2.9])
                with cert_col1:
                    st.markdown(f"""
                    <div style="background: white; border: 2px solid {cbfc_bg}; border-radius: 14px; padding: 1.1rem; text-align: center; box-shadow: 0 4px 14px rgba(0,0,0,0.06);">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #555; text-transform: uppercase;">CBFC Certificate (India)</div>
                        <div style="display: inline-block; background: {cbfc_bg}; color: white; font-size: 2.2rem; font-weight: 850; padding: 0.2rem 1.4rem; border-radius: 10px; margin: 0.6rem 0;">{cbfc_code}</div>
                        <div style="font-size: 0.82rem; font-weight: 600; color: #333;">{parents_guide['cbfc_rating']}</div>
                        <hr style="margin: 0.6rem 0; border: none; border-top: 1px dashed #ddd;">
                        <div style="font-size: 0.8rem; color: #666;">Global Equivalent: <span style="background: {mpaa_bg}; color: white; font-weight: 700; padding: 2px 8px; border-radius: 4px;">{mpaa_code}</span> ({parents_guide['mpaa_rating']})</div>
                    </div>
                    """, unsafe_allow_html=True)
                with cert_col2:
                    st.markdown(f"""
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px; padding: 1.1rem; height: 100%;">
                        <div style="font-size: 0.95rem; font-weight: 750; color: #1e293b; margin-bottom: 0.35rem;">IMDb Parents Guide & Content Advisory</div>
                        <div style="font-size: 0.85rem; color: #475569; line-height: 1.4; margin-bottom: 0.75rem;"><b>Reason for Rating:</b> {advisory_summary}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    cat_cols = st.columns(5)
                    cat_names = [
                        ('violence', 'Violence & Gore'),
                        ('profanity', 'Profanity & Slurs'),
                        ('adult_themes', 'Sex & Nudity'),
                        ('substances', 'Alcohol & Drugs'),
                        ('psychological_intensity', 'Distress / Intensity')
                    ]
                    for idx, (cat_key, cat_title) in enumerate(cat_names):
                        cat_info = parents_guide['categories'][cat_key]
                        sev = cat_info['severity']
                        cnt = cat_info['count']
                        sev_color = "#d32f2f" if sev == "Severe" else ("#f57c00" if sev == "Moderate" else ("#fbc02d" if sev == "Mild" else "#2e7d32"))
                        with cat_cols[idx]:
                            st.markdown(f"""
                            <div style="background: white; border: 1px solid #cbd5e1; border-top: 4px solid {sev_color}; border-radius: 8px; padding: 0.5rem; text-align: center;">
                                <div style="font-size: 0.72rem; font-weight: 600; color: #64748b; height: 1.8rem;">{cat_title}</div>
                                <div style="font-size: 0.92rem; font-weight: 750; color: {sev_color}; margin: 0.2rem 0;">{sev}</div>
                                <div style="font-size: 0.72rem; color: #94a3b8;">{cnt} flag(s)</div>
                            </div>
                            """, unsafe_allow_html=True)

                if parents_guide['character_breakdown']:
                    with st.expander("👤 Character Content Responsibility (Kaunsa character kitna offensive/violent hai)", expanded=False):
                        st.caption("Dialogues mein violence, slurs aur intense content use karne wale characters:")
                        char_df = pd.DataFrame(parents_guide['character_breakdown'])
                        st.dataframe(char_df, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.subheader("Story Dialogue & Pacing Signals")

                metric1, metric2, metric3, metric4 = st.columns(4)
                metric1.metric("Dialogue lines", f"{len(df):,}")
                metric2.metric("Characters", df['Character'].nunique())
                metric3.metric("Words analysed", f"{int(df['Word_Count'].sum()):,}")
                metric4.metric("Hesitation markers", int(df['Hesitation_Markers'].sum()))

                st.caption("Score signals: lexical variety, how dialogue is distributed across characters, and question/turn-taking frequency.")
                st.dataframe(pd.DataFrame([signal_components]), use_container_width=True, hide_index=True)

                st.subheader("Automatic screenplay sections")
                section_summary = df.groupby('Script_Section', observed=False).agg(
                    Dialogue_lines=('Dialogue', 'count'),
                    Words=('Word_Count', 'sum'),
                    Hesitation_markers=('Hesitation_Markers', 'sum'),
                    Average_words_per_line=('Word_Count', 'mean'),
                ).reset_index()
                section_summary['Share_of_dialogue_%'] = (section_summary['Dialogue_lines'] / len(df) * 100).round(1)
                section_summary['Share_of_words_%'] = (section_summary['Words'] / max(section_summary['Words'].sum(), 1) * 100).round(1)
                st.dataframe(section_summary.round(2), use_container_width=True, hide_index=True)
                if not is_hindi_script:
                    sentiment_by_section = (pd.crosstab(df['Script_Section'], df['Sentiment'], normalize='index') * 100).round(1).reset_index()
                    st.caption("Sentiment percentage by automatic section")
                    st.dataframe(sentiment_by_section, use_container_width=True, hide_index=True)
                st.caption("Sections are created automatically from script order. This works even when the source PDF has no usable scene headings.")

                download_col1, download_col2 = st.columns(2)
                with download_col1:
                    st.download_button(
                        "Download complete line-by-line report (CSV)",
                        data=df.to_csv(index=False).encode('utf-8'),
                        file_name="screenplay_complete_analysis.csv",
                        mime="text/csv",
                    )
                with download_col2:
                    st.download_button(
                        "Download section percentages (CSV)",
                        data=section_summary.to_csv(index=False).encode('utf-8'),
                        file_name="screenplay_section_summary.csv",
                        mime="text/csv",
                    )

                character_summary = df.groupby('Character').agg(
                    Lines=('Dialogue', 'count'),
                    Words=('Word_Count', 'sum'),
                    Avg_words_per_line=('Word_Count', 'mean'),
                    Questions=('Is_Question', 'sum'),
                    Hesitation_markers=('Hesitation_Markers', 'sum'),
                    Hesitations_per_100_words=('Hesitation_Rate_Per_100_Words', 'mean'),
                ).sort_values('Lines', ascending=False)
                st.subheader("Character breakdown")
                st.dataframe(character_summary.round(2), use_container_width=True)

            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    known_sentiments = {'Positive', 'Neutral', 'Negative'}
                    palette = {'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
                    active_sentiments = set(top_df['Sentiment'].astype(str).unique())
                    sns.countplot(data=top_df, y='Character', hue='Sentiment', ax=ax, palette=palette if active_sentiments.issubset(known_sentiments) else 'Blues')
                    ax.set_title("Sentiment Breakdown by Character")
                    st.pyplot(fig)
                with col2:
                    emotion_top_df = top_df[~top_df['Emotion'].isin(['Not sampled', 'Not sampled (CPU limit)', 'Language model required'])]
                    if emotion_top_df.empty:
                        st.info("Emotion heatmap needs an English or multilingual emotion model. The current PDF is being shown without invented emotion labels.")
                    else:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ct = pd.crosstab(emotion_top_df['Character'], emotion_top_df['Emotion'], normalize='index')
                        sns.heatmap(ct, cmap='Blues', annot=True, fmt=".2f", ax=ax)
                        ax.set_title("Emotion Proportion per Character (sampled lines)")
                        st.pyplot(fig)
                    
            with tab2:
                col1, col2 = st.columns(2)
                char_profiles = top_df.groupby('Character').agg({
                    'Word_Count': 'mean', 'Hesitation_Markers': 'sum', 'Dialogue': 'count',
                    'Hesitation_Rate_Per_100_Words': 'mean'
                })
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.barplot(x=char_profiles['Word_Count'], y=char_profiles.index, palette='viridis', ax=ax)
                    ax.set_title("Average Monologue Length (Words/Line)")
                    st.pyplot(fig)
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.barplot(x=char_profiles['Hesitation_Rate_Per_100_Words'], y=char_profiles.index, palette='Reds', ax=ax)
                    ax.set_title("Hesitation Markers per 100 Words")
                    st.pyplot(fig)
                st.caption("Counted markers: ..., --, um, uh, erm, and hmm. This is a speech-pattern signal, not a clinical or neurodivergence assessment.")
                hesitation_rows = df[df['Hesitation_Markers'] > 0][['Character', 'Scene', 'Dialogue', 'Hesitation_Markers']]
                if hesitation_rows.empty:
                    st.info("No hesitation markers were found in this screenplay. Try a script containing pauses (... / --) or fillers such as 'um' and 'uh'.")
                else:
                    st.subheader("Dialogue lines containing hesitation markers")
                    st.dataframe(hesitation_rows, use_container_width=True, hide_index=True)
            
            with tab3:
                if not df_zs.empty:
                    top_zs_chars = df_zs['Character'].value_counts().head(8).index
                    zs_plot_df = df_zs[df_zs['Character'].isin(top_zs_chars)]
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # Empathy Bar
                    char_empathy = zs_plot_df.groupby('Character')['Empathy_Score'].mean().sort_values()
                    colors = ['#e74c3c' if score < 0 else '#2ecc71' for score in char_empathy.values]
                    sns.barplot(x=char_empathy.values, y=char_empathy.index, palette=colors, ax=ax1)
                    ax1.axvline(0, color='black', linestyle='--')
                    ax1.set_title("Net Empathy Score")
                    
                    # Trait Stacked Bar
                    trait_counts = zs_plot_df.groupby(['Character', 'Dominant_Trait']).size().unstack(fill_value=0)
                    trait_percentages = trait_counts.div(trait_counts.sum(axis=1), axis=0) * 100
                    trait_percentages.plot(kind='bar', stacked=True, colormap='Spectral', ax=ax2)
                    ax2.set_title("Dominant Interpersonal Traits")
                    ax2.tick_params(axis='x', rotation=45)
                    
                    st.pyplot(fig)
                else:
                    if is_hindi_script:
                        st.warning("Interpersonal-tone results require a multilingual model for Hindi text; English-only predictions were skipped.")
                    else:
                        st.warning("Not enough substantial lines to run interpersonal-tone analysis.")

            with tab4:
                st.subheader("Search and inspect every extracted dialogue line")
                characters = ['All characters'] + sorted(df['Character'].unique().tolist())
                selected_character = st.selectbox("Character", characters)
                keyword = st.text_input("Search dialogue text")
                explorer_df = df.copy()
                if selected_character != 'All characters':
                    explorer_df = explorer_df[explorer_df['Character'] == selected_character]
                if keyword.strip():
                    explorer_df = explorer_df[explorer_df['Dialogue'].str.contains(keyword.strip(), case=False, na=False)]
                explorer_columns = ['Scene', 'Character', 'Dialogue', 'Sentiment', 'Emotion', 'Word_Count', 'Hesitation_Markers', 'Hesitation_Rate_Per_100_Words']
                st.dataframe(explorer_df[explorer_columns], use_container_width=True, hide_index=True)
                st.download_button(
                    "Download full analysis as CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name="screenplay_dialogue_analysis.csv",
                    mime="text/csv",
                )
            
            st.markdown("---")
            st.subheader("Raw Data Export")
            st.dataframe(df_zs if not df_zs.empty else df)
            
        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
