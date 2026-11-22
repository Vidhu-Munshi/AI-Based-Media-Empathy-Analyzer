"""
Emotion Detection Engine (100% Free, Rule-Based, Multilingual)
Detects 7 core emotions for each dialogue: Joy, Sadness, Anger, Fear, Disgust, Surprise, Neutral.
Works for English, Hindi, and Hinglish without any ML model download.
"""

import re
from typing import Dict, List

# Emotion keyword dictionaries (English + Hindi + Hinglish)
EMOTION_KEYWORDS = {
    'joy': [
        r'\b(?:happy|happiness|glad|joy|joyful|joyous|delighted|cheerful|wonderful|amazing|fantastic|love|loved|loving|lover|lovers|beloved|sweetheart|smile|smiling|laugh|laughing|laughter|celebrate|celebration|excited|excitement|grateful|thankful|thank\s*you|bliss|blessed|paradise|rejoice|rejoicing|hope|hopeful|pleasant|thrilled|cherish|kiss|kisses|rapture|fond|adore|adoring|ecstasy)\b',
        r'(?:खुश|खुशी|प्यार|मुस्कान|मुस्कुरा|हँसी|हँस|आनंद|उत्साह|धन्यवाद|शुक्रिया|मज़ा|मजा|सुंदर|अद्भुत|शानदार|जश्न|उम्मीद|प्रेम|दोस्ती|मोहब्बत|इश्क)',
        r'\b(?:khush|khushi|pyaar|pyar|muskaan|hasna|hassi|maza|mazaa|shukriya|dosti|mohabbat|ishq|umeed)\b',
    ],
    'sadness': [
        r'\b(?:sad|sadness|sorrow|sorrows|grief|griefs|grieve|mourning|mourn|mournful|cry|crying|cried|tears|weep|weeping|wept|depressed|depression|lonely|loneliness|alone|miss|missing|lost|heartbroken|broken|pain|painful|suffer|suffering|miserable|hopeless|despair|regret|regretful|sorry|apologize|unfortunate|tragic|tragedy|melancholy|gloomy|doom|woe|woeful|alas|sigh|sighs|lament|wretched|pity|pitiful|never\s*good\s*enough)\b',
        r'(?:दुख|दुखी|उदास|उदासी|रोना|रो|आँसू|अकेला|अकेली|अकेलापन|दर्द|तकलीफ|ग़म|गम|बेचारा|बेचारी|अफ़सोस|अफसोस|मायूस|मायूसी|बर्बाद|तबाह|विलाप)',
        r'\b(?:dukh|dukhi|udaas|rona|aansu|akela|dard|taklif|gam|mayus|barbad|tabah)\b',
    ],
    'anger': [
        r'\b(?:angry|anger|furious|fury|rage|raging|mad|hate|hatred|hating|hostile|hostility|violent|violence|fight|fighting|fought|destroy|destruction|revenge|vengeance|threatening|threaten|threat|curse|cursing|cursed|damn|hell|shut\s*up|get\s*out|stupid|idiot|fool|foolish|bastard|bloody|outraged|fed\s*up|resent|resentment|wrath|cruel|cruelty|brutal|brute|bully|foe|foes|enemy|enemies|quarrel|quarrels|brawl|brawling|strike|draw|spite|villain|villains|scoundrel|coward|cowards|scorn|plague)\b',
        r'(?:गुस्सा|क्रोध|नफ़रत|नफरत|घृणा|दुश्मन|दुश्मनी|बदला|धमकी|मार|मारना|पीटना|साला|साले|हरामी|हरामजादे|कमीने|कुत्ते|चुप|बाहर\s*निकल|जाहिल|उन्माद|आक्रोश)',
        r'\b(?:gussa|nafrat|dushman|badla|dhamki|saala|saale|harami|kamine|chup|jaahil)\b',
    ],
    'fear': [
        r'\b(?:afraid|fear|feared|fearful|scared|scary|terrified|terror|horrified|horror|panic|panicking|dread|dreading|dreadful|nightmare|anxious|anxiety|nervous|nervously|worried|worry|worrying|trembling|shaking|shock|shocked|alarmed|startled|helpless|trapped|danger|dangerous|run|flee|escape|hiding|paranoid|fright|frightened|peril|perilous|hazard|tremble|shiver|shivering|beware)\b',
        r'(?:डर|डरा|डरी|भय|भयभीत|खौफ|खौफनाक|दहशत|घबरा|घबराहट|चिंता|चिंतित|काँपना|काँप|सहम|बेबस|फंस|खतरा|खतरनाक|भाग)',
        r'\b(?:dar|dara|dari|bhay|khauf|dahshat|ghabra|chinta|kaamp|bebus|khatarnak|bhaag)\b',
    ],
    'disgust': [
        r'\b(?:disgust|disgusted|disgusting|gross|revolting|repulsive|nasty|filthy|filth|sick|sickening|vile|loathe|loathing|repelled|abhorrent|contempt|contemptuous|pathetic|shameful|shame|shameless|despise|despicable|abominable|horrid|horrible|hideous|wretched|detest|scorn|foul|pestilence)\b',
        r'(?:घिन|घृणा|शर्म|शर्मनाक|बेशर्म|घिनौना|गंदा|गंदी|नीच|कलंक|लज्जा|तिरस्कार|अपमान|नफ़रत|तौबा|छिः|थू)',
        r'\b(?:ghin|ghrina|sharm|besharam|ganda|gandi|neech|kalank|apamaan|taubaa)\b',
    ],
    'surprise': [
        r'\b(?:surprise|surprised|surprising|shock|shocked|shocking|astonished|astonishing|amazed|amazing|unbelievable|incredible|unexpected|sudden|suddenly|wow|whoa|gasp|gasped|stunned|dumbfounded|bewildered|awestruck|really|seriously|impossible|miracle|revelation|reveal|discovered|oh\s*my\s*god|what\s*the|marvel|behold|hark)\b',
        r'(?:हैरान|हैरानी|अचम्भा|अचरज|चौंक|चौंका|चकित|अविश्वसनीय|अचानक|क्या|सच|सचमुच|अरे|ओह|वाह|बाप\s*रे)',
        r'\b(?:hairan|achambha|chaunk|achanak|kya|sach|sachmuch|arey|wah|baap\s*re)\b',
    ],
}

def detect_emotion(text: str) -> Dict[str, float]:
    """
    Detect emotions in a text line. Returns a dict of emotion -> score (0.0 to 1.0).
    The dominant emotion gets the highest score. If no emotion keywords are found, 'neutral' = 1.0.
    """
    if not isinstance(text, str) or not text.strip():
        return {'joy': 0, 'sadness': 0, 'anger': 0, 'fear': 0, 'disgust': 0, 'surprise': 0, 'neutral': 1.0}

    lower = text.lower()
    scores = {}
    total_hits = 0

    for emotion, patterns in EMOTION_KEYWORDS.items():
        hits = 0
        for pat in patterns:
            hits += len(re.findall(pat, lower))
        scores[emotion] = hits
        total_hits += hits

    if total_hits == 0:
        scores = {e: 0.0 for e in EMOTION_KEYWORDS}
        scores['neutral'] = 1.0
        return scores

    # Normalize scores
    for e in scores:
        scores[e] = round(scores[e] / max(total_hits, 1), 3)
    scores['neutral'] = 0.0

    return scores


def detect_dominant_emotion(text: str) -> str:
    """Return the single dominant emotion label for a dialogue line."""
    scores = detect_emotion(text)
    return max(scores, key=scores.get)


def compute_character_emotions(df) -> Dict[str, Dict]:
    """
    Compute per-character emotion profiles from a dialogue DataFrame.
    Returns a dict of character -> { emotion_distribution, dominant_emotion, dialogue_count, ... }
    """
    import pandas as pd

    if df.empty or 'Dialogue' not in df.columns or 'Character' not in df.columns:
        return {}

    result = {}
    for char, group in df.groupby('Character'):
        emotion_totals = {e: 0.0 for e in list(EMOTION_KEYWORDS.keys()) + ['neutral']}
        line_emotions = []

        for _, row in group.iterrows():
            line_scores = detect_emotion(str(row['Dialogue']))
            dominant = max(line_scores, key=line_scores.get)
            line_emotions.append(dominant)
            for e, s in line_scores.items():
                emotion_totals[e] += s

        n = len(group)
        emotion_avg = {e: round(v / max(n, 1), 3) for e, v in emotion_totals.items()}

        # Emotion distribution (how many lines have each dominant emotion)
        from collections import Counter
        distribution = dict(Counter(line_emotions))

        # Character dominant persona:
        # A character is assigned an active emotion (e.g. Joy, Anger, Sadness) if:
        # 1. That active emotion has at least 3 lines (or >= 25% of lines in short roles)
        # 2. It represents the clear leader among active emotions (at least 35% of all active lines)
        # 3. Active lines make up at least 16% of their total dialogue lines.
        # Otherwise, they are predominantly Neutral / Analytical!
        active_counts = {e: distribution.get(e, 0) for e in EMOTION_KEYWORDS}
        top_active_emotion = max(active_counts, key=active_counts.get)
        top_active_count = active_counts[top_active_emotion]
        total_active_lines = sum(active_counts.values())

        min_lines = 1 if n <= 2 else (2 if n < 10 else 3)
        if top_active_count >= min_lines and \
           (top_active_count / max(total_active_lines, 1) >= 0.35) and \
           (top_active_count / max(n, 1) >= 0.15):
            dominant_overall = top_active_emotion
        else:
            dominant_overall = 'neutral'

        result[str(char)] = {
            'dialogue_count': n,
            'emotion_averages': emotion_avg,
            'dominant_emotion': dominant_overall,
            'emotion_distribution': distribution,
        }

    return result
