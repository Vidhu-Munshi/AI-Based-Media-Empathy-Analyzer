"""Schemas for FastAPI contracts."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class DialogueRequest(BaseModel):
    dialogue: str = Field(min_length=1, max_length=2000)
    character: Optional[str] = Field(default=None, max_length=100)


class DialogueFeaturesResponse(BaseModel):
    dialogue: str
    character: Optional[str]
    features: Dict[str, float | int]
    content_triggers: Dict[str, int]
    emotion: str
    emotion_scores: Dict[str, float]
    mode: str = "features_and_content_analysis"


class CategorySeverity(BaseModel):
    count: int
    severity: str


class CharacterBreakdown(BaseModel):
    Character: str
    Total_Lines: int
    Total_Content_Flags: int
    Violence_Flags: int
    Profanity_Flags: int
    Adult_Flags: int
    Substance_Flags: int
    Intensity_Flags: int


class ParentsGuideResponse(BaseModel):
    cbfc_code: str
    cbfc_rating: str
    mpaa_code: str
    mpaa_rating: str
    summary: str
    categories: Dict[str, CategorySeverity]
    character_breakdown: List[CharacterBreakdown]


class CharacterEmotionProfile(BaseModel):
    dialogue_count: int
    emotion_averages: Dict[str, float]
    dominant_emotion: str
    emotion_distribution: Dict[str, int]


class ScriptDoctorSuggestion(BaseModel):
    category: str
    tag: str
    icon: str
    issue: str
    suggestion: str


class DialogueItem(BaseModel):
    Movie: str
    Scene: str
    Character: str
    Dialogue: str


class ScreenplayAnalysisResponse(BaseModel):
    movie_title: str
    total_dialogues: int
    unique_characters: int
    total_words: int
    parents_guide: ParentsGuideResponse
    character_emotions: Dict[str, CharacterEmotionProfile]
    script_doctor_advice: List[ScriptDoctorSuggestion]
    characters: List[str]
    sample_dialogues: List[DialogueItem]
    mode: str = "full_screenplay_analysis"


class HealthResponse(BaseModel):
    status: str
    fastapi: bool = True
