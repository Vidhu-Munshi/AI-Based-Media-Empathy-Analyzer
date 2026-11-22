from model.kruti_to_unicode import is_kruti_dev, kruti_to_unicode
from model.inference import parse_screenplay_text
from model.content_advisory import compute_parents_guide
import pandas as pd


def test_kruti_dev_detection_and_conversion():
    kruti_sample = "?kj okilh\nf=kHkqou % flj tkos rks tkos & esjk oSfnd èkeZ u tkos ---"
    assert is_kruti_dev(kruti_sample) is True
    
    converted = kruti_to_unicode(kruti_sample)
    assert "घर वापसी" in converted
    assert "त्रिभुवन" in converted
    assert "वैदिक" in converted


def test_hindi_screenplay_parsing():
    script = """
पहला दृश्य

त्रिभुवन : सिर जावे तो जावे, धर्म न जावे।

हुस्ना : कहाँ जा रहे हो?

काजी : ठहर जाओ।
"""
    df = parse_screenplay_text(script, "test_play")
    assert not df.empty
    assert len(df) == 3
    assert list(df["Character"]) == ["त्रिभुवन", "हुस्ना", "काजी"]
    assert "सिर जावे तो जावे" in df.iloc[0]["Dialogue"]


def test_parents_guide_cbfc_rating():
    df_violent = pd.DataFrame([
        {"Character": "मुखिया", "Dialogue": "बाहर निकालो साले कसिया को! मार डालो इसको!"},
        {"Character": "काजी", "Dialogue": "तेजाब डाल देंगे इसके चेहरे पर, जिंदा जला देंगे।"},
        {"Character": "हुस्ना", "Dialogue": "छोड़ दो, मत मारो उसको!"}
    ])
    result = compute_parents_guide(df_violent)
    assert result["cbfc_code"] in ("A", "U/A 16+")
    assert result["categories"]["violence"]["count"] >= 3
    assert result["categories"]["profanity"]["count"] >= 1
    assert result["character_breakdown"][0]["Character"] in ("मुखिया", "काजी")
