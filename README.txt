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
├── README.md                       # Markdown version of project documentation
├── .gitignore                      # Git ignore file
├── 01_Data_Preprocessing.ipynb     # Notebook: screenplay parsing & cleaning
├── 02_Sentiment_Emotion.ipynb      # Notebook: VADER + emotion classification
├── 03_Lexical_Features.ipynb       # Notebook: communication-style features
├── 04_Inclusivity_Scoring.ipynb    # Notebook: zero-shot empathy scoring
├── data/
│   └── processed/                  # Pre-computed CSVs from notebook runs
├── wordcloud_masks/                # Mask images for word cloud generation
├── .streamlit/
│   └── config.toml                 # Streamlit server configuration
├── AI_Sample_Screenplay.txt        # Sample screenplay for testing
├── AMERICAN_PSYCHO.txt             # American Psycho screenplay
├── SW_EpisodeIV.txt                # Star Wars: A New Hope
├── SW_EpisodeV.txt                 # Star Wars: The Empire Strikes Back
└── SW_EpisodeVI.txt                # Star Wars: Return of the Jedi


##🛠️ Tech Stack
-------------

- Frontend: Streamlit
- NLP: NLTK, HuggingFace Transformers
- Models: VADER, DistilRoBERTa Emotion, BART-Large MNLI
- Visualization: Matplotlib, Seaborn
