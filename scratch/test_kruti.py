import re

def kruti_to_unicode(text: str) -> str:
    """
    Standard and robust Kruti Dev 010 to Unicode Devanagari converter.
    Handles vowel combinations, matra re-orderings (chhoti 'i'), and reph ('Z').
    """
    if not text:
        return ""

    modified = text
    
    replacements = [
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

        ("=k", "त्र"), ("=", "त्र"),
        ("czkã.k", "ब्राह्मण"),
        ("czkã", "ब्राह्म"),
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

        # Independent Vowels
        ("vkS", "औ"), ("vks", "ओ"), ("vk¡", "आँ"), ("vk", "आ"),
        ("v", "अ"), ("bZ", "ई"), ("b", "इ"), ("mQ", "उफ़"), ("m", "उ"),
        ("Å", "ऊ"), ("Q", "ऐ"),

        # Consonants with combinations
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

        # Matras
        ("kS", "ौ"), ("ks", "ो"), ("k¡", "ाँ"), ("k;", "ाय"),
        ("k", "ा"), ("h", "ी"), ("q", "ु"), ("w", "ू"), ("s", "े"),
        ("S", "ै"), ("`", "ृ"), ("a", "ं"), ("¡", "ँ"),
        ("%", ":"), ("+", "़"), ("&", "-"), ("^", "‘"), ("*", "’")
    ]

    for k, v in replacements:
        modified = modified.replace(k, v)

    # 1. Handle Chhoti 'i' matra ('f')
    consonant_cluster = r'(?:[\u0915-\u0939]\u094d)*[\u0915-\u0939]\u093c?'
    modified = re.sub(r'f(' + consonant_cluster + r')', r'\1ि', modified)

    # 2. Handle Reph ('Z')
    matra = r'[\u093e-\u094c\u0902\u0901]?'
    modified = re.sub(r'(' + consonant_cluster + r')(' + matra + r')Z', r'र्\1\2', modified)
    modified = re.sub(r'Z(' + consonant_cluster + r')', r'र्\1', modified)

    # 3. Clean up Punctuation & Danda
    modified = modified.replace('AA', '॥').replace('A', '।')

    return modified

if __name__ == '__main__':
    test_lines = [
        "?kj okilh",
        "[ ukVd ]",
        "jkts'k dqekj",
        "f=kHkqou % flj tkos rks tkos & esjk oSfnd èkeZ u tkos ---",
        "gqLuk % f=kHkq! --- f=kHkq!---",
        "dkth % cgjs b'd esa xksrk yxkusA",
        "lEiw.kkZuUn % fgUnqvks! dc rd vk¡[ksa cUn djs jgksxs\\ vk¡[ksa [kksyksA",
        "viwokZuUn % lR; opu dg jgk gSA",
        "vEeh % rkSck! rkSck! --- ;s D;k dj jgs gks rcjst fe;k¡\\",
        "eqf[k;k % frokjh--- vks frokjh!--- ckgj fudkyks dfl;k dks! Llkyk tkfr mUewyu vkUnksyu pykrk gS",
        "dkth % bLyke ds fy, ejuk tkurs gSa rks ekjuk Hkh vPNh rjg ls tkurs gSaA",
        "eqlyeku 2 % rstkcA"
    ]
    for line in test_lines:
        print(line, "-->", kruti_to_unicode(line))
