from model.emotion_detector import detect_emotion, detect_dominant_emotion, compute_character_emotions
from model.inference import parse_screenplay_text
import pandas as pd


def test_detect_emotion_keywords():
    joy_line = "I am so happy and delighted to see you, thank you!"
    scores = detect_emotion(joy_line)
    assert scores['joy'] > 0
    assert detect_dominant_emotion(joy_line) == 'joy'

    anger_line = "I hate you, shut up you bloody bastard!"
    scores_anger = detect_emotion(anger_line)
    assert scores_anger['anger'] > 0
    assert detect_dominant_emotion(anger_line) == 'anger'

    neutral_line = "The door is on the left side."
    scores_neutral = detect_emotion(neutral_line)
    assert scores_neutral['neutral'] == 1.0


def test_hindi_emotion_detection():
    hindi_joy = "मुझे बहुत खुशी हुई आपसे मिलकर, बहुत प्यार करते हैं आपसे।"
    assert detect_dominant_emotion(hindi_joy) == 'joy'

    hindi_anger = "गुस्सा मत दिला मुझे, चुप रह साले!"
    assert detect_dominant_emotion(hindi_anger) == 'anger'


def test_character_emotions_pipeline():
    df = pd.DataFrame([
        {'Character': 'HERO', 'Dialogue': 'I love this wonderful life!'},
        {'Character': 'HERO', 'Dialogue': 'We are so blessed and happy.'},
        {'Character': 'VILLAIN', 'Dialogue': 'I will destroy you with hatred!'},
        {'Character': 'VILLAIN', 'Dialogue': 'Get out before I attack!'}
    ])
    result = compute_character_emotions(df)
    assert 'HERO' in result
    assert 'VILLAIN' in result
    assert result['HERO']['dominant_emotion'] == 'joy'
    assert result['VILLAIN']['dominant_emotion'] == 'anger'
