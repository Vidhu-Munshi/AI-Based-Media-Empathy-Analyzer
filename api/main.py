"""
FastAPI Backend for Screenplay Media Empathy & Censor Rating Analyzer.
100% Free, high-speed, local processing.
Serves both a modern interactive Web UI and full REST API endpoints.
"""

import os
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    DialogueFeaturesResponse,
    DialogueRequest,
    HealthResponse,
    ScreenplayAnalysisResponse,
)
from model.content_advisory import analyze_line_content, compute_parents_guide
from model.emotion_detector import detect_emotion, detect_dominant_emotion, compute_character_emotions
from model.script_doctor import generate_script_doctor_advice
from model.inference import clean_dialogue_text, extract_lexical_features, parse_screenplay_text
from model.kruti_to_unicode import is_kruti_dev, kruti_to_unicode

app = FastAPI(
    title="AI Screenplay Empathy & Censor Rating Analyzer",
    description="Universal Screenplay Parsing, Character Comparison, Emotion Detection, CBFC/IMDb Parents Guide, and AI Script Doctor Advice.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract selectable text from PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="pypdf is required for PDF parsing.") from exc

    try:
        reader = PdfReader(BytesIO(file_bytes))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages_text).strip()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF file: {exc}") from exc


@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Serve the modern interactive single-page Web UI."""
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    return HTMLResponse("<h1>FastAPI Screenplay Analyzer running. Visit <a href='/docs'>/docs</a></h1>")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", fastapi=True)


@app.post("/api/analyze-dialogue", response_model=DialogueFeaturesResponse)
async def analyze_dialogue(request: DialogueRequest) -> DialogueFeaturesResponse:
    dialogue = clean_dialogue_text(request.dialogue)
    if not dialogue:
        raise HTTPException(status_code=400, detail="Dialogue has no analyzable text")

    lexical = extract_lexical_features(request.dialogue)
    triggers = analyze_line_content(request.dialogue)
    emotion_scores = detect_emotion(request.dialogue)
    dominant = detect_dominant_emotion(request.dialogue)

    return DialogueFeaturesResponse(
        dialogue=dialogue,
        character=request.character,
        features=lexical,
        content_triggers=triggers,
        emotion=dominant,
        emotion_scores=emotion_scores,
    )


@app.post("/api/analyze-screenplay", response_model=ScreenplayAnalysisResponse)
async def analyze_screenplay(file: UploadFile = File(...)) -> ScreenplayAnalysisResponse:
    filename = file.filename or "unknown.txt"
    suffix = Path(filename).suffix.lower()

    if suffix not in (".txt", ".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .txt or .pdf file.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    movie_title = Path(filename).stem

    # Extract text based on file format
    if suffix == ".pdf":
        text_content = extract_text_from_pdf(raw_bytes)
    else:
        text_content = raw_bytes.decode("utf-8", errors="ignore")

    # Auto-convert Kruti Dev legacy Hindi if detected
    if is_kruti_dev(text_content):
        text_content = kruti_to_unicode(text_content)

    if not text_content.strip():
        raise HTTPException(status_code=400, detail="No readable text found in screenplay.")

    # Parse screenplay into dialogue dataframe
    dialogues_df = parse_screenplay_text(text_content, movie_title)
    if dialogues_df.empty:
        raise HTTPException(
            status_code=400,
            detail="No character dialogues could be extracted. Ensure the file has character names like 'CHARACTER:' or 'पात्र : संवाद'."
        )

    # Compute content advisory & age ratings
    parents_guide = compute_parents_guide(dialogues_df)

    # Compute per-character emotion profiles
    character_emotions = compute_character_emotions(dialogues_df)

    # Generate AI Script Doctor suggestions
    script_doctor_advice = generate_script_doctor_advice(dialogues_df, parents_guide, character_emotions)

    # Calculate total words
    total_words = int(dialogues_df["Dialogue"].apply(lambda d: len(d.split())).sum())
    characters = sorted(list(dialogues_df["Character"].unique()))

    # Sample dialogues for explorer (up to 500)
    sample_dialogues = dialogues_df.head(500).to_dict("records")

    return ScreenplayAnalysisResponse(
        movie_title=movie_title,
        total_dialogues=len(dialogues_df),
        unique_characters=len(characters),
        total_words=total_words,
        parents_guide=parents_guide,
        character_emotions=character_emotions,
        script_doctor_advice=script_doctor_advice,
        characters=characters,
        sample_dialogues=sample_dialogues,
    )
