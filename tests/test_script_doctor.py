import pytest
import pandas as pd
from model.script_doctor import generate_script_doctor_advice

def test_script_doctor_dominance_advice():
    # Character A dominates > 55%
    data = {
        'Character': ['ALICE'] * 8 + ['BOB'] * 2,
        'Dialogue': ['Hello there!'] * 10
    }
    df = pd.DataFrame(data)
    parents_guide = {'cbfc_code': 'U', 'categories': {}}
    emotions = {'ALICE': {'emotion_distribution': {'neutral': 8}}, 'BOB': {'emotion_distribution': {'neutral': 2}}}
    
    advice = generate_script_doctor_advice(df, parents_guide, emotions)
    assert any(a['category'] == 'Character Balance' for a in advice)
    assert any('ALICE' in a['issue'] for a in advice)

def test_script_doctor_censor_advice():
    # CBFC 'A' rating triggers advice on lowering to U/A
    data = {
        'Character': ['JOHN', 'MIKE'],
        'Dialogue': ['I will kill you bastard', 'Run now']
    }
    df = pd.DataFrame(data)
    parents_guide = {
        'cbfc_code': 'A',
        'categories': {
            'violence': {'severity': 'Severe', 'count': 5},
            'profanity': {'severity': 'Moderate', 'count': 3}
        }
    }
    emotions = {}
    
    advice = generate_script_doctor_advice(df, parents_guide, emotions)
    assert any(a['category'] == 'Censor / Audience Reach' for a in advice)
    assert any('Violence & Gore' in a['issue'] for a in advice)

def test_script_doctor_monologue_advice():
    long_line = "Word " * 50
    data = {
        'Character': ['SPEAKER'] * 5,
        'Dialogue': [long_line] * 5
    }
    df = pd.DataFrame(data)
    parents_guide = {'cbfc_code': 'U', 'categories': {}}
    emotions = {}
    
    advice = generate_script_doctor_advice(df, parents_guide, emotions)
    assert any(a['category'] == 'Pacing & Rhythm' for a in advice)
