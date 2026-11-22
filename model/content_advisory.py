"""
IMDb Parents Guide & CBFC (Central Board of Film Certification) Rating Engine
100% Free, rule-based, deterministic NLP analysis for screenplays (English, Hindi, Hinglish).
Analyzes dialogue content across 5 standard Parents Guide categories:
1. Violence & Gore
2. Profanity & Abusive Language
3. Sex & Nudity / Adult Themes
4. Alcohol, Drugs & Smoking
5. Psychological Distress & Intense Conflict
"""

import re
from typing import Dict, List, Tuple, Any
import pandas as pd

# Multi-lingual keyword dictionaries & regex patterns
CONTENT_PATTERNS = {
    'violence': [
        # English: Weapons, lethal actions, and physical violence (excluding common dramatic/poetic words like "burn" or "hang")
        r'\b(?:kill|killed|killing|murder|murdered|murderer|stab|stabbed|shoot|shot|gun|guns|knife|knives)\b',
        r'\b(?:blood|bloody|corpse|corpses|lynch|lynching|strangle|strangled|torture|behead|slaughter)\b',
        r'\b(?:acid\s*attack|burn\s*alive|burned\s*alive|hanged|lynched|slain|execution|assault|lethal|poison|poisoned|dagger|sword)\b',
        # Hindi & Hinglish
        r'(?:हत्या|मौत|मार\s*डाल|लाश|खून|तेजाब|जिंदा\s*जला|फांसी|चाकू|भाले|बरछी|काटकर|लटका\s*दिया)',
        r'(?:धक्का|मार-पीट|पीट|गोली|जहर|दहशतगर्दी|कत्ल|दंगा|खून-खराबा|मृत्युदण्ड|हलाल|हमला)',
        r'\b(?:maar|jaan\s*se\s*maar|qatl|lash|khoon|hathya|tezaab|jalaa|phaansi|danga|goli)\b'
    ],
    'profanity': [
        # English
        r'\b(?:fuck|fucking|fucker|fucked|shit|bitch|bastard|bastards|asshole|assholes|cunt|dick|piss|damn|hell)\b',
        # Hindi & Hinglish
        r'(?:साले|साला|हरामजादे|हरामी|कुत्ते|कमीने|रांड|भड़वे|भड़ुआ|मादरचोद|बहनचोद|चूतिया|गांड|औकात|नालहक|काफिर|म्लेच्छ|सूअर)',
        r'\b(?:saale|saala|haramzada|harami|kutte|kamine|bhadwe|chutiya|gaand|aukaat|nalahaq|kafir|chamar|bhadva)\b'
    ],
    'adult_themes': [
        # English
        r'\b(?:sex|sexual|nude|nudity|naked|breasts|orgasm|penis|vagina|whore|prostitute|porn|intercourse)\b',
        # Hindi & Hinglish
        r'(?:यौन|नंगा|नग्न|बलात्कार|संभोग|वेश्या|रंडी|जिना|हवस|कामुक|जिस्म|कोठा|अय्याशी)',
        r'\b(?:jism|hawas|balatkar|randi|kotha|nanga|sambhog|zina)\b'
    ],
    'substances': [
        # English
        r'\b(?:alcohol|wine|whiskey|vodka|beer|drugs|cocaine|heroin|marijuana|weed|smoke|smoking|cigarette|drunk)\b',
        # Hindi & Hinglish
        r'(?:शराब|दारू|मदिरा|चरस|गांजा|अफीम|हेरोइन|सिगरेट|बीड़ी|हुक्का|नशा|नशेड़ी|शराबी)',
        r'\b(?:sharab|daru|madira|charas|ganja|afeem|nasha|nashe|cigarette|bidi)\b'
    ],
    'psychological_intensity': [
        # English
        r'\b(?:terror|terrorize|panic|screaming|nightmare|horrific|trauma|insanity|insane|depressed|suicide)\b',
        # Hindi & Hinglish
        r'(?:आत्महत्या|दहशत|खौफ|चीख|चीत्कार|सहम|उन्माद|आतंक|पागलपन|विद्रोह|सन्नाटा|कांपती|कलंक)',
        r'\b(?:aatmahatya|dahshat|khauf|cheekh|aatamk|pagalpan|dar|seham)\b'
    ]
}

# Explicit adult triggers for distinguishing CBFC 'A' (18+) from U/A 13+ and U/A 16+
GRAPHIC_GORE_PATTERNS = [
    r'\b(?:slaughter|behead|decapitat\w*|tortur\w*|mutilat\w*|massacre|lynch\w*|strangl\w*|butcher\w*|bloodbath|dismember\w*)\b',
    r'(?:हत्या|जिंदा\s*जला|तेजाब|फांसी|काटकर\s*फेंक|कत्लेआम|गला\s*घोंट|चीर|फाड़)'
]

HARD_PROFANITY_PATTERNS = [
    r'\b(?:fuck|fucking|fucker|fucked|motherfucker|cunt|cocksucker)\b',
    r'(?:मादरचोद|बहनचोद|चूतिया|गांड|भड़वे|रांड|chutiya|gaand|bhadwe)'
]

def analyze_line_content(text: str) -> Dict[str, int]:
    """Check a single line of dialogue for triggers in each category."""
    if not isinstance(text, str) or not text.strip():
        return {cat: 0 for cat in CONTENT_PATTERNS}

    results = {}
    lower_text = text.lower()
    for cat, patterns in CONTENT_PATTERNS.items():
        count = 0
        for pat in patterns:
            matches = re.findall(pat, lower_text)
            count += len(matches)
        results[cat] = count
    return results

def compute_severity(count: int, cat: str, total_lines: int = 100) -> str:
    """Determine None, Mild, Moderate, or Severe based on triggers and script volume."""
    if count == 0:
        return "None"
    
    rate = count / max(total_lines, 1)

    if cat == 'violence':
        if total_lines < 20:
            if rate >= 0.5 or count >= 5:
                return "Severe"
            elif rate >= 0.25 or count >= 2:
                return "Moderate"
            return "Mild"
        else:
            if count <= 8 and rate < 0.03:
                return "Mild"
            elif count <= 40 or rate < 0.10:
                return "Moderate"
            else:
                return "Severe"

    elif cat == 'adult_themes':
        if count <= 2 or rate < 0.01:
            return "Mild"
        elif count <= 7 or rate < 0.035:
            return "Moderate"
        else:
            return "Severe"

    elif cat == 'profanity':
        if total_lines < 20:
            if rate >= 0.4 or count >= 4:
                return "Severe"
            elif rate >= 0.2 or count >= 2:
                return "Moderate"
            return "Mild"
        else:
            if count <= 5 or rate < 0.02:
                return "Mild"
            elif count <= 25 or rate < 0.07:
                return "Moderate"
            else:
                return "Severe"

    else:  # substances, psychological_intensity
        if count <= 6 or rate < 0.03:
            return "Mild"
        elif count <= 30 or rate < 0.09:
            return "Moderate"
        else:
            return "Severe"


def compute_parents_guide(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive IMDb Parents Guide & CBFC certification rating
    from the screenplay dialogue dataframe.
    """
    if df.empty or 'Dialogue' not in df.columns:
        return {
            'cbfc_rating': 'U',
            'mpaa_rating': 'G',
            'summary': 'Insufficient dialogue found for age certification.',
            'categories': {cat: {'count': 0, 'severity': 'None'} for cat in CONTENT_PATTERNS},
            'character_breakdown': []
        }

    total_lines = len(df)

    # Line-by-line scanning
    line_scores = []
    char_triggers = {}

    for _, row in df.iterrows():
        diag = str(row.get('Dialogue', ''))
        char = str(row.get('Character', 'UNKNOWN')).strip()
        scores = analyze_line_content(diag)
        line_scores.append(scores)

        if char not in char_triggers:
            char_triggers[char] = {cat: 0 for cat in CONTENT_PATTERNS}
            char_triggers[char]['total_lines'] = 0

        char_triggers[char]['total_lines'] += 1
        for cat, cnt in scores.items():
            char_triggers[char][cat] += cnt

    # Aggregate movie level
    movie_counts = {cat: sum(s[cat] for s in line_scores) for cat in CONTENT_PATTERNS}
    categories = {}
    for cat, cnt in movie_counts.items():
        categories[cat] = {
            'count': cnt,
            'severity': compute_severity(cnt, cat, total_lines)
        }

    # Count graphic gore and hard profanity occurrences across the screenplay
    full_text = " ".join(df['Dialogue'].dropna().astype(str)).lower()
    graphic_gore_count = sum(len(re.findall(pat, full_text)) for pat in GRAPHIC_GORE_PATTERNS)
    hard_profanity_count = sum(len(re.findall(pat, full_text)) for pat in HARD_PROFANITY_PATTERNS)

    # Determine CBFC & MPAA Certificate
    v_sev = categories['violence']['severity']
    p_sev = categories['profanity']['severity']
    a_sev = categories['adult_themes']['severity']
    i_sev = categories['psychological_intensity']['severity']
    s_sev = categories['substances']['severity']

    reasons = []

    # Check for CBFC 'A' (Adults Only 18+) / MPAA 'R' criteria:
    # Requires genuine graphic gore/torture, pervasive hard profanity, or explicit adult sexual content
    is_adult_rating = False
    if a_sev == 'Severe':
        is_adult_rating = True
        reasons.append("Explicit sexual content or adult themes")
    elif graphic_gore_count >= 8 or (v_sev == 'Severe' and graphic_gore_count >= 4):
        is_adult_rating = True
        reasons.append("Extreme graphic gore, sadistic violence, or lethal brutality")
    elif hard_profanity_count >= 15 or (p_sev == 'Severe' and hard_profanity_count >= 8):
        is_adult_rating = True
        reasons.append("Pervasive hard abusive language and severe slurs")
    elif v_sev == 'Severe' and p_sev == 'Severe':
        is_adult_rating = True
        reasons.append("Pervasive combination of extreme violence and heavy abusive language")

    if is_adult_rating:
        cbfc = "A"
        cbfc_label = "A (Adults Only 18+)"
        mpaa = "R"
        mpaa_label = "R (Restricted 17+)"
    elif v_sev in ('Severe', 'Moderate') or p_sev == 'Moderate' or i_sev == 'Severe' or graphic_gore_count >= 1:
        # Moderate-to-high action tension / drama (e.g. Romeo & Juliet, The Dark Knight, Gladiator)
        # Properly maps to U/A 16+ or U/A 13+
        if v_sev == 'Severe' or i_sev == 'Severe' or graphic_gore_count >= 1 or categories['violence']['count'] >= 14:
            cbfc = "U/A 16+"
            cbfc_label = "U/A 16+ (Parental Guidance 16+)"
            mpaa = "PG-13"
            mpaa_label = "PG-13 (Parents Strongly Cautioned)"
            reasons.append("Strong dramatic conflict, weapon duels, or thematic distress")
        else:
            cbfc = "U/A 13+"
            cbfc_label = "U/A 13+ (Parental Guidance 13+)"
            mpaa = "PG-13"
            mpaa_label = "PG-13 (Some Material May Be Inappropriate)"
            reasons.append("Moderate action violence, mild language, or thematic intensity")
    elif 'Mild' in [v_sev, p_sev, i_sev, s_sev]:
        cbfc = "U/A 7+"
        cbfc_label = "U/A 7+ (Parental Guidance 7+)"
        mpaa = "PG"
        mpaa_label = "PG (Parental Guidance Suggested)"
        reasons.append("Mild thematic elements or fantasy peril")
    else:
        cbfc = "U"
        cbfc_label = "U (Universal - All Ages)"
        mpaa = "G"
        mpaa_label = "G (General Audiences - All Ages)"
        reasons.append("Clean, wholesome dialogue suitable for all viewers")

    # Character breakdown ranking
    char_list = []
    for char, data in char_triggers.items():
        total_flags = sum(data[cat] for cat in CONTENT_PATTERNS)
        char_list.append({
            'Character': char,
            'Total_Lines': data['total_lines'],
            'Total_Content_Flags': total_flags,
            'Violence_Flags': data['violence'],
            'Profanity_Flags': data['profanity'],
            'Adult_Flags': data['adult_themes'],
            'Substance_Flags': data['substances'],
            'Intensity_Flags': data['psychological_intensity'],
        })

    char_list = sorted(char_list, key=lambda x: (x['Total_Content_Flags'], x['Total_Lines']), reverse=True)

    return {
        'cbfc_code': cbfc,
        'cbfc_rating': cbfc_label,
        'mpaa_code': mpaa,
        'mpaa_rating': mpaa_label,
        'summary': "; ".join(reasons) if reasons else "Dialogue cleared across all advisory categories.",
        'categories': categories,
        'character_breakdown': char_list
    }
