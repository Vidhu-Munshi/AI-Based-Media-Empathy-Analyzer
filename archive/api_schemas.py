"""
Pydantic Schemas for API Request/Response Validation
=====================================================
Shared schemas used across the FastAPI application.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


# ============================================================
# Enums
# ============================================================

class EmotionLabel(str, Enum):
    """Supported emotion labels (20 classes)."""
    # Basic emotions (Ekman)
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    JOY = "joy"
    SADNESS = "sadness"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"

    # Social emotions (Plutchik + custom)
    ANTICIPATION = "anticipation"
    TRUST = "trust"
    CONTEMPT = "contempt"
    EMBARRASSMENT = "embarrassment"
    PRIDE = "pride"
    GRATITUDE = "gratitude"

    # Empathy-specific
    EMPATHY = "empathy"
    HOSTILITY = "hostility"
    CONDESCENSION = "condescension"
    SUPPORTIVENESS = "supportiveness"
    DISMISSIVENESS = "dismissiveness"
    MOCKING = "mocking"
    RESPECTFUL = "respectful"


class StyleLabel(str, Enum):
    """Supported communication style labels (8 classes)."""
    MONOLOGUER = "monologuer"
    INQUISITIVE = "inquisitive"
    HESITANT = "hesitant"
    DIRECT = "direct"
    REPETITIVE = "repetitive"
    DIVERSE = "diverse"
    EMOTIONAL = "emotional"
    ANALYTICAL = "analytical"


class EmpathyLabel(str, Enum):
    """Qualitative empathy labels."""
    HIGHLY_EMPATHETIC = "Highly Empathetic"
    EMPATHETIC = "Empathetic"
    NEUTRAL = "Neutral"
    LOW_EMPATHY = "Low Empathy"
    HOSTILE = "Hostile"


# ============================================================
# Request Schemas
# ============================================================

class DialogueRequest(BaseModel):
    """Request for single dialogue analysis."""
    dialogue: str = Field(
        ...,
        description="Dialogue text to analyze",
        min_length=1,
        max_length=2000,
        example="I understand how you feel, and I'm here to support you."
    )
    character: Optional[str] = Field(
        None,
        description="Character name (optional)",
        max_length=100,
        example="DR. SMITH"
    )
    context: Optional[str] = Field(
        None,
        description="Previous dialogue for context (optional)",
        max_length=2000,
        example="I'm really worried about the test results."
    )

    @validator('dialogue')
    def dialogue_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Dialogue cannot be empty")
        return v.strip()


class ScreenplayAnalysisRequest(BaseModel):
    """Request for full screenplay analysis."""
    return_character_profiles: bool = Field(
        True,
        description="Include per-character aggregated profiles"
    )
    return_imdb_prediction: bool = Field(
        True,
        description="Include IMDb rating prediction"
    )
    max_lines: Optional[int] = Field(
        500,
        description="Maximum dialogue lines to process (for performance)",
        ge=1,
        le=5000
    )
    min_words_per_line: int = Field(
        3,
        description="Minimum words per line to analyze",
        ge=1,
        le=50
    )


class BatchDialogueRequest(BaseModel):
    """Request for batch dialogue analysis."""
    dialogues: List[DialogueRequest] = Field(
        ...,
        description="List of dialogues to analyze",
        min_items=1,
        max_items=100
    )


class CharacterProfileRequest(BaseModel):
    """Request for character profile aggregation."""
    character: str = Field(..., description="Character name")
    dialogues: List[DialogueRequest] = Field(
        ...,
        description="All dialogues by this character",
        min_items=1
    )


class IMDBPredictionRequest(BaseModel):
    """Request for IMDb rating prediction."""
    character_profiles: List["CharacterProfileResponse"] = Field(
        ...,
        description="Aggregated character profiles",
        min_items=1
    )
    movie_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional movie metadata (genre, year, etc.)"
    )


# ============================================================
# Response Schemas
# ============================================================

class EmotionScores(BaseModel):
    """Emotion probabilities for all 20 classes."""
    anger: float = Field(..., ge=0.0, le=1.0)
    disgust: float = Field(..., ge=0.0, le=1.0)
    fear: float = Field(..., ge=0.0, le=1.0)
    joy: float = Field(..., ge=0.0, le=1.0)
    sadness: float = Field(..., ge=0.0, le=1.0)
    surprise: float = Field(..., ge=0.0, le=1.0)
    neutral: float = Field(..., ge=0.0, le=1.0)
    anticipation: float = Field(..., ge=0.0, le=1.0)
    trust: float = Field(..., ge=0.0, le=1.0)
    contempt: float = Field(..., ge=0.0, le=1.0)
    embarrassment: float = Field(..., ge=0.0, le=1.0)
    pride: float = Field(..., ge=0.0, le=1.0)
    gratitude: float = Field(..., ge=0.0, le=1.0)
    empathy: float = Field(..., ge=0.0, le=1.0)
    hostility: float = Field(..., ge=0.0, le=1.0)
    condescension: float = Field(..., ge=0.0, le=1.0)
    supportiveness: float = Field(..., ge=0.0, le=1.0)
    dismissiveness: float = Field(..., ge=0.0, le=1.0)
    mocking: float = Field(..., ge=0.0, le=1.0)
    respectful: float = Field(..., ge=0.0, le=1.0)

    def to_dict(self) -> Dict[str, float]:
        return self.dict()

    def top_k(self, k: int = 3) -> List[tuple]:
        """Get top k emotions by probability."""
        items = [(k, v) for k, v in self.dict().items()]
        return sorted(items, key=lambda x: -x[1])[:k]


class LexicalFeatures(BaseModel):
    """Lexical/communication features extracted from dialogue."""
    word_count: int = Field(..., ge=0)
    sentence_count: int = Field(..., ge=1)
    words_per_sentence: float = Field(..., ge=0.0)
    lexical_diversity_ttr: float = Field(..., ge=0.0, le=1.0)
    hesitation_markers: int = Field(..., ge=0)
    is_question: int = Field(..., ge=0, le=1)


class DialogueAnalysisResult(BaseModel):
    """Complete analysis result for a single dialogue."""
    dialogue: str
    character: Optional[str]
    emotions: EmotionScores
    top_emotions: List[str] = Field(..., min_items=1, max_items=5)
    empathy_score: float = Field(..., ge=-3.0, le=3.0)
    empathy_label: EmpathyLabel
    style: StyleLabel
    style_confidence: float = Field(..., ge=0.0, le=1.0)
    style_distribution: Dict[str, float] = Field(
        ...,
        description="Probability distribution over all 8 styles"
    )
    lexical_features: LexicalFeatures
    processing_time_ms: float = Field(..., ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DialogueResponse(BaseModel):
    """API response for single dialogue analysis."""
    success: bool = True
    data: DialogueAnalysisResult
    model_version: str = "1.0.0"


class BatchDialogueResponse(BaseModel):
    """API response for batch dialogue analysis."""
    success: bool = True
    data: List[DialogueAnalysisResult]
    count: int
    model_version: str = "1.0.0"
    total_processing_time_ms: float


class CharacterProfileResponse(BaseModel):
    """Aggregated character profile."""
    character: str
    total_lines: int = Field(..., ge=1)
    avg_empathy_score: float = Field(..., ge=-3.0, le=3.0)
    dominant_emotions: List[Dict[str, Any]] = Field(
        ...,
        description="Top emotions with counts and percentages"
    )
    communication_style: StyleLabel
    style_distribution: Dict[str, float] = Field(
        ...,
        description="Percentage distribution over all 8 styles"
    )
    verbosity: float = Field(..., description="Average words per line")
    lexical_diversity: float = Field(..., ge=0.0, le=1.0)
    question_rate: float = Field(..., ge=0.0, le=1.0)
    hesitation_rate: float = Field(..., ge=0.0)
    empathy_trend: Optional[List[float]] = Field(
        None,
        description="Empathy score progression through script (optional)"
    )


class EmotionDistribution(BaseModel):
    """Overall emotion distribution across screenplay."""
    counts: Dict[str, int]
    percentages: Dict[str, float]
    total_lines: int


class ScreenplayAnalysisResponse(BaseModel):
    """Complete screenplay analysis response."""
    success: bool = True
    movie_title: str
    total_dialogues: int
    unique_characters: int
    character_profiles: List[CharacterProfileResponse]
    overall_empathy_score: float = Field(..., ge=-3.0, le=3.0)
    emotion_distribution: EmotionDistribution
    imdb_prediction: Optional[float] = Field(None, ge=1.0, le=10.0)
    imdb_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    processing_time_ms: float
    model_version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IMDBPredictionResponse(BaseModel):
    """IMDb rating prediction response."""
    success: bool = True
    predicted_rating: float = Field(..., ge=1.0, le=10.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    contributing_factors: Dict[str, float] = Field(
        ...,
        description="Feature importance / contributing factors"
    )
    character_profiles_used: int
    model_version: str = "1.0.0"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    model_parameters: Optional[int] = None
    uptime_seconds: Optional[float] = None
    version: str = "1.0.0"


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    model_type: str
    architecture: str
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    max_seq_len: int
    num_emotions: int
    num_styles: int
    emotion_labels: List[str]
    style_labels: List[str]
    parameter_count: int
    training_config: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = False
    error: str
    error_code: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# Utility Functions
# ============================================================

def create_error_response(
    error: str,
    error_code: str = "INTERNAL_ERROR",
    details: Optional[Dict] = None,
    status_code: int = 500
) -> JSONResponse:
    """Create standardized error response."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=error,
            error_code=error_code,
            details=details
        ).dict()
    )


def validate_emotion_scores(scores: Dict[str, float]) -> bool:
    """Validate emotion scores dict."""
    required = set(e.value for e in EmotionLabel)
    provided = set(scores.keys())
    return required.issubset(provided) and all(0.0 <= v <= 1.0 for v in scores.values())


def validate_style_distribution(dist: Dict[str, float]) -> bool:
    """Validate style distribution dict."""
    required = set(s.value for s in StyleLabel)
    provided = set(dist.keys())
    return required.issubset(provided) and all(0.0 <= v <= 1.0 for v in dist.values())


# Forward reference resolution
IMDBPredictionRequest.update_forward_refs()