"""
Script Doctor / Script Consultant Advisor Engine
100% Free, local, rule-based & deterministic, with optional local LLM hook (Ollama).
Analyzes dialogue dynamics, pacing, character balance, emotion arcs, and content flags
to provide concrete, actionable editorial suggestions for improving the screenplay.
"""

from typing import Dict, List, Any
import pandas as pd


def generate_script_doctor_advice(df: pd.DataFrame, parents_guide: Dict[str, Any], character_emotions: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generate professional script doctor feedback and creative suggestions
    based on the screenplay analysis metrics.
    """
    suggestions = []
    total_lines = len(df)
    if total_lines == 0:
        return suggestions

    char_counts = df['Character'].value_counts()
    unique_chars = len(char_counts)
    top_char = char_counts.index[0]
    top_share = char_counts.iloc[0] / total_lines

    # 1. Dialogue Dominance & Screen-time balance
    if top_share > 0.55:
        suggestions.append({
            "category": "Character Balance",
            "tag": "High Dialogue Dominance",
            "icon": "⚖️",
            "issue": f"Character '{top_char}' dominates {int(top_share * 100)}% of all dialogue lines in the script.",
            "suggestion": f"Consider delegating exposition or counter-arguments to secondary characters. Giving scene partners more agency and rebuttal lines will increase conversational tension and avoid one-sided lecturing."
        })
    elif top_share < 0.25 and unique_chars > 8:
        suggestions.append({
            "category": "Protagonist Focus",
            "tag": "Scattered Dialogue Focus",
            "icon": "🎭",
            "issue": f"No single protagonist holds more than {int(top_share * 100)}% of the script's dialogue across {unique_chars} characters.",
            "suggestion": "The narrative may feel ensemble-heavy without a strong emotional anchor. Clarify whose journey the audience is following by strengthening the lead's dialogue presence."
        })

    # 2. Monologue length & Pacing
    df['word_count'] = df['Dialogue'].apply(lambda d: len(str(d).split()))
    long_speeches = df[df['word_count'] > 45]
    if len(long_speeches) >= 4:
        examples = ", ".join(long_speeches['Character'].head(2).unique())
        suggestions.append({
            "category": "Pacing & Rhythm",
            "tag": "Dense Monologues Detected",
            "icon": "⏱️",
            "issue": f"Found {len(long_speeches)} long monologues (>45 words) spoken by characters like {examples}.",
            "suggestion": "Break up dense blocks of dialogue with physical action beats, subtextual interruptions, or character reactions to keep the cinematic tempo dynamic."
        })

    # 3. Content Advisory & Age Rating optimization
    cbfc_code = parents_guide.get('cbfc_code', 'U')
    cats = parents_guide.get('categories', {})
    v_sev = cats.get('violence', {}).get('severity', 'None')
    p_sev = cats.get('profanity', {}).get('severity', 'None')

    if cbfc_code == 'A':
        triggers = []
        if v_sev in ('Severe', 'Moderate'):
            triggers.append(f"Violence & Gore ({v_sev})")
        if p_sev in ('Severe', 'Moderate'):
            triggers.append(f"Profanity & Slurs ({p_sev})")
        trigger_str = " and ".join(triggers) if triggers else "graphic content"

        suggestions.append({
            "category": "Censor / Audience Reach",
            "tag": "Adult (18+) Rating Trigger",
            "icon": "🎯",
            "issue": f"The script currently receives a CBFC 'A' rating primarily due to {trigger_str}.",
            "suggestion": "If you want a broader theatrical audience (U/A 13+ or U/A 16+), consider shifting violent acts into off-screen sound design/reactions, and toning down explicit slurs into psychological subtext."
        })

    # 4. Emotional Arc & Monotony check
    em_totals = {'joy': 0, 'sadness': 0, 'anger': 0, 'fear': 0, 'disgust': 0, 'surprise': 0, 'neutral': 0}
    for char, data in character_emotions.items():
        dist = data.get('emotion_distribution', {})
        for e, cnt in dist.items():
            if e in em_totals:
                em_totals[e] += cnt

    active_emotions = sum(em_totals[e] for e in ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise'])
    neutral_ratio = em_totals['neutral'] / max(total_lines, 1)

    if em_totals['anger'] > active_emotions * 0.55:
        suggestions.append({
            "category": "Emotional Dynamics",
            "tag": "Sustained High Hostility",
            "icon": "🔥",
            "issue": "Anger and hostility make up over 55% of all emotionally charged dialogue.",
            "suggestion": "Constant shouting or confrontation can exhaust the audience. Insert moments of vulnerability, hesitation, or brief relief to create contrast, making dramatic climaxes hit harder."
        })
    elif em_totals['joy'] < 3 and total_lines > 20:
        suggestions.append({
            "category": "Emotional Range",
            "tag": "Lack of Comic Relief / Warmth",
            "icon": "🌤️",
            "issue": "The screenplay has almost zero joyful, empathetic, or lighthearted beats.",
            "suggestion": "Introduce small bonding moments or gallows humor between allies. Even dark dramas (like American Psycho or Godfather) use irony to heighten audience engagement."
        })

    # 5. Question & Curiosity Rate (Subtext vs Direct Answers)
    question_lines = df[df['Dialogue'].str.contains(r'\?|क्या|क्यों|कहाँ|किस|who|what|why|where|how', case=False, na=False)]
    q_rate = len(question_lines) / total_lines
    if q_rate < 0.12 and total_lines > 15:
        suggestions.append({
            "category": "Subtext & Conflict",
            "tag": "Low Inquisitiveness (Expository Dialogue)",
            "icon": "💡",
            "issue": f"Only {int(q_rate * 100)}% of lines are questions. Characters mostly state facts directly to each other.",
            "suggestion": "Great dialogue often uses questions, deflection, and unsaid subtext rather than direct statements. Have characters challenge each other with probing questions instead of passive listening."
        })

    # Fallback suggestion if script is already well balanced
    if len(suggestions) < 3:
        suggestions.append({
            "category": "Dialogue Polish",
            "tag": "Subtext & Physicality",
            "icon": "✨",
            "issue": "Dialogue flow is clean and structurally balanced.",
            "suggestion": "Focus on enhancing subtext: allow characters to say one thing while their physical actions (parentheticals) imply the opposite for cinematic depth."
        })

    return suggestions
