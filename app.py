import os
import re
import tempfile
import warnings
from typing import List, Dict, Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
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

@st.cache_resource(show_spinner="Loading ML Models into memory (this happens once)...")
def load_models():
    """Downloads NLTK resources and caches HuggingFace pipelines."""
    # NLTK setup
    for resource in ['vader_lexicon', 'punkt', 'punkt_tab']:
        try:
            if resource == 'vader_lexicon':
                nltk.data.find(f'sentiment/{resource}.zip')
            else:
                nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)
            
    sia = SentimentIntensityAnalyzer()
    
    # Device configuration (GPU if available)
    device = 0 if torch.cuda.is_available() else -1
    
    # HuggingFace Pipelines
    emotion_clf = pipeline(
        "text-classification", 
        model="j-hartmann/emotion-english-distilroberta-base", 
        device=device,
        truncation=True,
        max_length=512
    )
    
    zero_shot_clf = pipeline(
        "zero-shot-classification", 
        model="facebook/bart-large-mnli", 
        device=device
    )
    
    return sia, emotion_clf, zero_shot_clf

sia, emotion_classifier, zs_classifier = load_models()

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

def clean_character_name(line: str) -> str:
    cleaned = re.sub(r'\(.*?\)', '', line).strip()
    return re.sub(r'[^A-Z0-9\s\-/]', '', cleaned).strip()

def clean_dialogue_text(text: str) -> str:
    text = re.sub(r'\(.*?\)', '', text).lower()
    text = re.sub(r'[^a-z0-9\s\.,!\?\'\"]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def parse_screenplay(file_path: str) -> pd.DataFrame:
    movie_name = os.path.basename(file_path).split('.')[0]
    extracted_data = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

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
            if is_page_number_or_copyright(stripped):
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

def extract_lexical_features(text: str) -> dict:
    words, sentences = word_tokenize(str(text)), sent_tokenize(str(text))
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    unique_words = set(w.lower() for w in words if w.isalnum())
    
    return {
        'Word_Count': word_count,
        'Lexical_Diversity_TTR': len(unique_words) / word_count if word_count > 0 else 0,
        'Hesitation_Markers': len(re.findall(r'\.\.\.|--', str(text))),
        'Is_Question': 1 if '?' in str(text) else 0
    }

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
st.title("🎬 AI-Based Media Empathy Analyzer")
st.markdown("Upload a screenplay to analyze character communication styles, inclusivity, and neurodivergent mapping.")

with st.sidebar:
    st.header("Pipeline Configuration")
    sample_size = st.slider("Zero-Shot Sample Size (Lines)", min_value=100, max_value=2000, value=500, step=100, 
                            help="Limits the number of lines processed by BART-Large to prevent UI timeouts.")

uploaded_file = st.file_uploader("Upload Screenplay (.txt)", type=["txt"])

if uploaded_file is not None:
    if st.button("Run Full NLP Pipeline", type="primary"):
        progress_bar = st.progress(0)
        status = st.empty()
        
        try:
            # --- STEP 1: Parsing ---
            status.text("Step 1/4: Parsing screenplay structure...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
                
            df = parse_screenplay(tmp_path)
            os.remove(tmp_path)
            progress_bar.progress(25)
            
            if df.empty:
                st.error("No dialogue could be extracted. Please check the file formatting.")
                st.stop()
                
            # --- STEP 2: Sentiment & Emotion ---
            status.text("Step 2/4: Extracting Sentiment & Emotion profiles...")
            df['Vader'] = df['Dialogue'].apply(lambda x: sia.polarity_scores(x)['compound'])
            df['Sentiment'] = pd.cut(df['Vader'], bins=[-1, -0.05, 0.05, 1], labels=['Negative', 'Neutral', 'Positive'])
            
            emotions = emotion_classifier(df['Dialogue'].tolist())
            df['Emotion'] = [e['label'].capitalize() for e in emotions]
            progress_bar.progress(50)
            
            # --- STEP 3: Lexical Features ---
            status.text("Step 3/4: Calculating Lexical & Communication Features...")
            features_df = pd.DataFrame(df['Dialogue'].apply(extract_lexical_features).tolist())
            df = pd.concat([df, features_df], axis=1)
            progress_bar.progress(75)
            
            # --- STEP 4: Zero-Shot Inclusivity Scoring ---
            status.text(f"Step 4/4: Running Zero-Shot Classifier on top {sample_size} lines...")
            
            df_zs = df[df['Word_Count'] >= 5].sort_values(by='Word_Count', ascending=False).head(sample_size).copy()
            zs_results = pd.DataFrame(df_zs['Dialogue'].apply(score_inclusivity).tolist())
            df_zs = pd.concat([df_zs.reset_index(drop=True), zs_results], axis=1)
            
            progress_bar.progress(100)
            status.success("Analysis Complete!")
            
            # ==========================================
            # 4. DASHBOARD VISUALIZATIONS
            # ==========================================
            st.markdown("---")
            tab1, tab2, tab3 = st.tabs(["Emotional Landscape", "Communication Profiles", "Inclusivity & Empathy"])
            sns.set_theme(style="whitegrid")
            
            top_chars = df['Character'].value_counts().head(8).index
            top_df = df[df['Character'].isin(top_chars)]
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.countplot(data=top_df, y='Character', hue='Sentiment', ax=ax, palette={'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'})
                    ax.set_title("Sentiment Breakdown by Character")
                    st.pyplot(fig)
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ct = pd.crosstab(top_df['Character'], top_df['Emotion'], normalize='index')
                    sns.heatmap(ct, cmap='Blues', annot=True, fmt=".2f", ax=ax)
                    ax.set_title("Emotion Proportion per Character")
                    st.pyplot(fig)
                    
            with tab2:
                col1, col2 = st.columns(2)
                char_profiles = top_df.groupby('Character').agg({
                    'Word_Count': 'mean', 'Hesitation_Markers': 'sum', 'Dialogue': 'count'
                })
                char_profiles['Hesitation_Rate'] = char_profiles['Hesitation_Markers'] / char_profiles['Dialogue']
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.barplot(x=char_profiles['Word_Count'], y=char_profiles.index, palette='viridis', ax=ax)
                    ax.set_title("Average Monologue Length (Words/Line)")
                    st.pyplot(fig)
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.barplot(x=char_profiles['Hesitation_Rate'], y=char_profiles.index, palette='Reds', ax=ax)
                    ax.set_title("Hesitation/Masking Rate per Line")
                    st.pyplot(fig)
            
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
                    st.warning("Not enough complex lines to run Zero-Shot analysis.")
            
            st.markdown("---")
            st.subheader("Raw Data Export")
            st.dataframe(df_zs if not df_zs.empty else df)
            
        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")