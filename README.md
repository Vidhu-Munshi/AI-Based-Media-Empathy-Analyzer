##🎬 AI-Based Media Empathy Analyzer
===================================

An NLP-powered Streamlit dashboard that analyzes movie screenplays to measure how empathetic, inclusive, and emotionally nuanced character dialogue is.

Upload a `.txt` screenplay and the app runs a 4-step ML pipeline that extracts dialogue, profiles sentiment & emotion, computes communication-style features, and scores each line for inclusivity and empathy using zero-shot classification.


##✨ Features
----------

1. Screenplay Parsing
   - Extracts structured dialogue (character + lines) from raw screenplay text.
   - Powered by a custom regex parser.

2. Sentiment & Emotion
   - Scores each line's sentiment and classifies its dominant emotion.
   - Powered by VADER + DistilRoBERTa.

3. Lexical Features
   - Measures word count, vocabulary diversity (TTR), hesitation markers, and question frequency.
   - Powered by NLTK tokenizers.

4. Inclusivity Scoring
   - Zero-shot classification against traits like empathetic, supportive, respectful vs. condescending, dismissive, mocking.
   - Computes a net Empathy Score.
   - Powered by BART-Large MNLI.


#Dashboard Tabs
--------------

- Tab 1: Emotional Landscape
  Sentiment breakdowns and emotion heatmaps per character.

- Tab 2: Communication Profiles
  Average monologue length, hesitation/masking rates.

- Tab 3: Inclusivity & Empathy
  Net empathy scores and dominant interpersonal traits.


##🚀 Getting Started
------------------

Prerequisites:
- Python 3.10+
- (Optional) CUDA-compatible GPU for faster inference

Installation:
1. Clone this repository to your local machine.
2. Create and activate a virtual environment:
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
3. Install dependencies:
   pip install streamlit pandas numpy matplotlib seaborn nltk transformers torch

Run the App:
   streamlit run app.py


##📂 Project Structure
--------------------

├── app.py                          # Streamlit dashboard (main entry point)
├── README.txt                      # This project documentation file
├── .gitignore                      # Git ignore file
├── model/                          # Notebook-based preprocessing and analysis
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_Sentiment_Emotion.ipynb
│   ├── 03_Lexical_Features.ipynb
│   └── 04_Inclusivity_Scoring.ipynb
├── data/                          # Processed CSV outputs from notebook runs
│   └── processed/
├── data for tranning/             # Raw screenplay source files used for training/testing
│   ├── AMERICAN_PSYCHO.txt
│   ├── SW_EpisodeIV.txt
│   ├── SW_EpisodeV.txt
│   └── SW_EpisodeVI.txt
├── testing/                       # Extra screenplay samples and test files
│   ├── AI_Sample_Screenplay.txt
│   ├── Ishaan_Script_Extracted.txt
│   └── Original_All_Caps_Dialogue_Script.txt
├── cache/                         # Local model and tokenizer cache directories
│   └── huggingface/
├── wordcloud_masks/               # Mask images for word cloud generation
└── .streamlit/
    └── config.toml                # Streamlit server configuration


##🛠️ Tech Stack
-------------

- Frontend: Streamlit
- NLP: NLTK, HuggingFace Transformers
- Models: VADER, DistilRoBERTa Emotion, BART-Large MNLI
- Visualization: Matplotlib, Seaborn
