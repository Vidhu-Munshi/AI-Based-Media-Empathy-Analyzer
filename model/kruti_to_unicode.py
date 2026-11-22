"""
Kruti Dev 010 to Devanagari Unicode Converter & Detector
100% Free, deterministic, offline, zero-dependency string converter.
Converts legacy Kruti Dev / Remington font encoded text to standard Unicode Hindi.
"""

import re
from typing import Tuple

# Mapping of Kruti Dev multi-character and single-character glyphs to Unicode
KRUTI_DEV_MAPPING = [
    # Complex ligatures & special symbols
    ("ñ", "ह्र"), ("ò", "द्द"), ("ó", "द्ध"), ("ô", "द्ब"), ("õ", "द्म"),
    ("ö", "द्य"), ("÷", "द्व"), ("§", "ऽ"), ("°", "त्त"), ("¶", "फ़"),
    ("¼", "("), ("½", ")"), ("¿", "?"), ("À", "!"), ("Á", "।"), ("Â", "॥"),
    ("Í", "क्ष"), ("Î", "ज्ञ"), ("Ï", "द्र"), ("Ð", "प्र"), ("Ñ", "श्र"),
    ("Ò", "त्र"), ("Ó", "ष्ट्र"), ("Ô", "ष्ट"), ("Õ", "ष्ठ"), ("Ö", "ष्ठ"),
    ("×", "द्ग"), ("Ø", "द्घ"), ("Ù", "द्ध"), ("Ú", "द्ब"), ("Û", "द्भ"),
    ("Ü", "द्म"), ("Ý", "द्य"), ("Þ", "द्व"), ("ß", "श्च"), ("à", "ह्न"),
    ("á", "ह्म"), ("â", "ह्य"), ("ã", "ह्व"), ("ä", "ह्ल"), ("å", "ऋ"),
    ("æ", "क्त"), ("ç", "क्र"), ("è", "ध"), ("é", "ञ"), ("ê", "ट्ट"),
    ("ë", "ट्ठ"), ("ì", "ड्ड"), ("í", "ड्ढ"), ("î", "कृ"), ("ï", "दृ"),
    ("ð", "पृ"),

    # Common combinations
    ("=k", "त्र"), ("=", "त्र"),
    ("czkã.k", "ब्राह्मण"), ("czkã", "ब्राह्म"),
    ("Lokeh", "स्वामी"),
    ("lEiw.kkZuUn", "सम्पूर्णानन्द"),
    ("viwokZuUn", "अपूर्वानन्द"),
    ("f=kHkqou", "त्रिभुवन"),
    ("f=kHkq", "त्रिभु"),
    ("rcjst", "तबरेज"),
    ("gqLuk", "हुस्ना"),
    ("dkth", "काजी"),
    ("eqf[k;k", "मुखिया"),
    ("dfl;k", "कसिया"),
    ("Nk;k", "छाया"),
    ("dk'khukFk", "काशीनाथ"),
    ("vkSjr", "औरत"),
    ("vèksM+", "अधेड़"),
    ("yM+dk", "लड़का"),
    ("yM+ds", "लड़के"),
    ("lkèkq", "साधु"),
    ("eU=kh", "मन्त्री"),
    ("ea=kh", "मंत्री"),
    ("eqlyeku", "मुसलमान"),
    ("vEeh", "अम्मी"),
    ("xzkeh.k", "ग्रामीण"),
    ("èkeZ", "धर्म"),
    ("deZ", "कर्म"),
    ("o.kZ", "वर्ण"),
    ("iw.kZ", "पूर्ण"),

    # Independent Vowels
    ("vkS", "औ"), ("vks", "ओ"), ("vk¡", "आँ"), ("vk", "आ"),
    ("v", "अ"), ("bZ", "ई"), ("b", "इ"), ("mQ", "उफ़"), ("m", "उ"),
    ("Å", "ऊ"), ("Q", "ऐ"),

    # Consonants with multi-char representations
    ("[k", "ख"), ("?k", "घ"), ("Fk", "थ"), ("èk", "ध"),
    ("Hk", "भ"), ("'k", "श"), ('"k', "ष"),

    # Single consonants & half-consonants
    ("D", "क्"), ("d", "क"), ("[", "ख्"), ("x", "ग"), ("X", "ग्"),
    ("?", "घ"), ("p", "च"), ("P", "च्"), ("N", "छ"), ("t", "ज"),
    ("T", "ज्"), (">", "झ"), ("V", "ट"), ("B", "ठ"), ("M", "ड"),
    ("<", "ढ"), ("r", "त"), ("R", "त्"), ("F", "थ्"), ("n", "द"),
    ("u", "न"), ("U", "न्"), ("i", "प"), ("I", "प्"), ("c", "ब"),
    ("C", "ब्"), ("H", "भ्"), ("e", "म"), ("E", "म्"), (";", "य"),
    ("Y", "य्"), ("j", "र"), ("y", "ल"), ("Y", "ल्"), ("o", "व"),
    ("O", "व्"), ("'", "श्"), ('"', "ष्"), ("l", "स"), ("L", "स्"),
    ("g", "ह"),

    # Matras & modifiers
    ("kS", "ौ"), ("ks", "ो"), ("k¡", "ाँ"), ("k;", "ाय"),
    ("k", "ा"), ("h", "ी"), ("q", "ु"), ("w", "ू"), ("s", "े"),
    ("S", "ै"), ("`", "ृ"), ("a", "ं"), ("¡", "ँ"),
    (",", "ए"), ("%", ":"), ("+", "़"), ("&", "-"),
    ("^", "‘"), ("*", "’")
]

def is_kruti_dev(text: str) -> bool:
    """
    Detect whether an input text (or PDF extracted text) is in Kruti Dev encoding.
    Returns True if typical Kruti Dev Hindi tokens appear frequently and English dictionary share is near zero.
    """
    if not text or len(text.strip()) < 20:
        return False
        
    # If the text already has substantial Devanagari, it's not Kruti Dev ASCII
    devanagari_chars = len(re.findall(r'[\u0900-\u097f]', text))
    if devanagari_chars > len(text) * 0.15:
        return False

    sample = text[:4000]
    words = re.findall(r"[A-Za-z'?~%]+", sample)
    if len(words) < 10:
        return False

    # Common Kruti Dev token signatures
    kruti_markers = {
        'gS', 'gSa', 'dks', 'esa', 'dh', 'dk', 'ds', 'us', 'Fkk', 'Fkh', 'Fks',
        'rkS', 'bl', 'ml', 'vkSj', 'fQj', 'ugha', 'dsoy', 'le>k', 'vius', 'viuh',
        '?kj', 'okilh', 'f=k', 'czkã', 'gqLuk', 'rcjst', 'dkth', 'yM+dk'
    }
    
    kruti_hits = sum(w in kruti_markers or any(w.startswith(m) for m in ('f=', 'f=k', 'czk', 'vèk', 'lEi', 'viw')) for w in words)
    kruti_ratio = kruti_hits / max(len(words), 1)

    # English stopword check to avoid false positives on legitimate English texts
    english_stopwords = {
        'the', 'and', 'that', 'this', 'with', 'from', 'you', 'your', 'have',
        'are', 'was', 'for', 'not', 'but', 'all', 'they', 'she', 'him', 'her'
    }
    english_hits = sum(w.lower() in english_stopwords for w in words)
    english_ratio = english_hits / max(len(words), 1)

    # Kruti Dev detection rule
    return kruti_ratio > 0.05 or (english_ratio < 0.02 and any(m in sample for m in ['?kj', 'f=kHkqou', 'gqLuk', 'czkã', 'lEiw.kkZ']))

def kruti_to_unicode(text: str) -> str:
    """
    Convert a Kruti Dev encoded string to standard Unicode Devanagari Hindi.
    """
    if not text:
        return ""

    modified = text

    # Apply substitutions
    for kruti, devanagari in KRUTI_DEV_MAPPING:
        modified = modified.replace(kruti, devanagari)

    # 1. Handle Chhoti 'i' matra ('f')
    # In Kruti Dev, 'f' precedes consonant/cluster. In Unicode it attaches as 'ि' after.
    consonant_cluster = r'(?:[\u0915-\u0939]\u094d)*[\u0915-\u0939]\u093c?'
    modified = re.sub(r'f(' + consonant_cluster + r')', r'\1ि', modified)

    # 2. Handle Reph ('Z')
    # In Kruti Dev, 'Z' follows the letter. In Unicode it is 'र्' before the consonant.
    matra = r'[\u093e-\u094c\u0902\u0901]?'
    modified = re.sub(r'(' + consonant_cluster + r')(' + matra + r')Z', r'र्\1\2', modified)
    modified = re.sub(r'Z(' + consonant_cluster + r')', r'र्\1', modified)

    # 3. Clean up Punctuation & Viram
    modified = modified.replace('AA', '॥').replace('A', '।')

    return modified
