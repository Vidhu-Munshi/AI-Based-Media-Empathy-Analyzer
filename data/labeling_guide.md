# 📋 Data Labeling Guide: AI-Based Media Empathy Analyzer (Milestone 1)

> **Scope**: Compact schema for Milestone 1 — 7 emotions × 5-point empathy × uncertain flag.
> **Format**: JSONL (one dialogue per line).
> **Target**: 2,000+ gold-labeled lines minimum.

---

## 🎯 Label Schema (Milestone 1)

Each dialogue receives **exactly three annotations**:

| Field | Type | Values | Required |
|-------|------|--------|----------|
| `emotion` | Single-label | `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, `neutral` | ✅ |
| `empathy` | Ordinal 5-point | `-2`, `-1`, `0`, `+1`, `+2` | ✅ |
| `uncertain` | Boolean | `true` / `false` | ✅ |

**Additional metadata** (auto-filled or annotator-provided):
- `dialogue` — cleaned text
- `character` — speaker name
- `movie` — source screenplay
- `scene` — scene heading
- `source` — `gold` (human) or `weak_label` (auto-generated)
- `annotator` — initials
- `timestamp` — ISO 8601

---

## 📝 1. Emotion Definitions (7 Classes — Mutually Exclusive)

| Label | Definition | Key Indicators |
|-------|------------|----------------|
| `anger` | Hostility, frustration, aggression | Insults, threats, raised voice, "!", aggressive verbs |
| `disgust` | Revulsion, contempt, moral rejection | "sick", "pathetic", "gross", moral superiority tone |
| `fear` | Anxiety, worry, apprehension | "scared", "worried", "afraid", hesitation markers (`...`, `--`) |
| `joy` | Happiness, amusement, pleasure | Laughter, positive exclamations, praise, humor |
| `sadness` | Grief, disappointment, melancholy | "sorry", "unfortunate", loss references, subdued tone |
| `surprise` | Shock, unexpectedness | "what?", "no way", "really?", sudden topic shifts |
| `neutral` | Factual, informational, no affect | Pure exposition, technical descriptions, procedural talk |

**Rule**: Pick the **single dominant emotion**. If truly mixed, choose the one with strongest linguistic evidence. If genuinely ambiguous → `uncertain: true`.

---

## 📊 2. Empathy Scale (5-Point Ordinal)

| Score | Label | Definition | Positive Traits | Negative Traits |
|-------|-------|------------|-----------------|-----------------|
| `+2` | Highly Empathetic | Deep understanding, validation, emotional resonance | `empathetic`, `supportive`, `respectful` | — |
| `+1` | Supportive | Encouraging, helpful, acknowledges feelings | `supportive`, `respectful` | — |
| `0` | Neutral | Factual, transactional, no emotional engagement | — | — |
| `-1` | Low Empathy | Dismissive, minimizing, emotionally distant | — | `dismissive`, `condescending` |
| `-2` | Hostile | Actively antagonistic, invalidating, aggressive | — | `hostile`, `mocking`, `condescending` |

**Scoring Guide**:
- **+2**: "I completely understand how hard this is for you. Your feelings are valid."
- **+1**: "That sounds difficult. Let me know if you need anything."
- **0**: "The meeting is at 3 PM. Bring the report."
- **-1**: "You're overreacting. It's not a big deal."
- **-2**: "Whatever. You're being dramatic and wrong."

**Rule**: Assign the **single best score**. Do not interpolate (no `-1.5`). If between two scores, pick the stronger evidence. If genuinely ambiguous → `uncertain: true`.

---

## ❓ 3. Uncertain Flag

Set `uncertain: true` when:
- Annotator confidence < 0.7
- Dialogue is too short (< 5 words) to judge
- Genuine ambiguity between two emotions or empathy scores
- Sarcasm/irony detected but intent unclear
- Cultural/contextual references unknown

**Effect**: Uncertain samples are excluded from primary training metrics but retained for analysis.

---

## 📁 4. Output Format: `data/labeled/dialogues.jsonl`

One JSON object per line:

```json
{
  "dialogue": "i understand how hard this is for you and i want you to know that your feelings are completely valid",
  "character": "EMMA",
  "movie": "AI_SAMPLE_SCREENPLAY",
  "scene": "INT. CITY PARK - SUNSET",
  "emotion": "sadness",
  "empathy": 2,
  "uncertain": false,
  "source": "gold",
  "meta": {
    "annotator": "VH",
    "timestamp": "2025-11-21T18:30:00",
    "notes": "Validating Aaron's feelings about not being good enough"
  }
}
```

### Additional annotated examples

These examples demonstrate the remaining emotion labels and the difference between
emotion and interpersonal empathy. Each is a separate JSONL record when used in a dataset.

```json
{"dialogue":"get away from me right now","character":"RAVI","movie":"EXAMPLE","scene":"INT. HALL - DAY","emotion":"anger","empathy":-2,"uncertain":false,"source":"gold"}
{"dialogue":"that smell is revolting","character":"RAVI","movie":"EXAMPLE","scene":"INT. HALL - DAY","emotion":"disgust","empathy":0,"uncertain":false,"source":"gold"}
{"dialogue":"i am scared that we will not make it","character":"LEAH","movie":"EXAMPLE","scene":"EXT. ROAD - NIGHT","emotion":"fear","empathy":0,"uncertain":false,"source":"gold"}
{"dialogue":"we did it this is wonderful","character":"LEAH","movie":"EXAMPLE","scene":"EXT. ROAD - NIGHT","emotion":"joy","empathy":1,"uncertain":false,"source":"gold"}
{"dialogue":"i miss her every day","character":"RAVI","movie":"EXAMPLE","scene":"INT. HOME - NIGHT","emotion":"sadness","empathy":0,"uncertain":false,"source":"gold"}
{"dialogue":"you are here already","character":"LEAH","movie":"EXAMPLE","scene":"INT. HOME - NIGHT","emotion":"surprise","empathy":0,"uncertain":false,"source":"gold"}
{"dialogue":"please enter your employee number","character":"SYSTEM","movie":"EXAMPLE","scene":"INT. OFFICE - DAY","emotion":"neutral","empathy":0,"uncertain":false,"source":"gold"}
```

```json
{
  "dialogue": "whatever youre being dramatic and wrong",
  "character": "AARON",
  "movie": "AI_SAMPLE_SCREENPLAY",
  "scene": "INT. CITY PARK - SUNSET",
  "emotion": "anger",
  "empathy": -2,
  "uncertain": false,
  "source": "gold",
  "meta": {
    "annotator": "VH",
    "timestamp": "2025-11-21T18:31:00",
    "notes": "Dismissive and hostile response to validation"
  }
}
```

```json
{
  "dialogue": "the meeting is at 3 pm bring the report",
  "character": "MANAGER",
  "movie": "OFFICE_SCRIPT",
  "scene": "INT. OFFICE - DAY",
  "emotion": "neutral",
  "empathy": 0,
  "uncertain": false,
  "source": "gold",
  "meta": {
    "annotator": "VH",
    "timestamp": "2025-11-21T18:32:00",
    "notes": "Purely transactional"
  }
}
```

### **Required Fields** (validation enforced)
- `dialogue` (string, 1-2000 chars, cleaned lowercase)
- `character` (string, 1-100 chars)
- `movie` (string, 1-100 chars)
- `scene` (string, 1-200 chars)
- `emotion` (string, one of 7 values)
- `empathy` (integer, one of `[-2, -1, 0, 1, 2]`)
- `uncertain` (boolean)
- `source` (string, `gold` or `weak_label`)

### **Optional Fields**
- `meta.annotator` (string, 2-4 chars)
- `meta.timestamp` (ISO 8601)
- `meta.notes` (string, free text)

---

## 🛠️ 5. Annotation Workflow

### **Tool: Label Studio (Recommended)**
```bash
pip install label-studio
label-studio start
# Open http://localhost:8080
# Create project → Import labeling_guide_milestone1.xml (see below)
# Annotate → Export JSONL
```

### **Label Studio Config (`labeling_guide_milestone1.xml`)**

```xml
<View>
  <Text name="dialogue" value="$dialogue" />
  
  <Choices name="emotion" toName="dialogue" choice="single" showInLine="true">
    <Choice value="anger" />
    <Choice value="disgust" />
    <Choice value="fear" />
    <Choice value="joy" />
    <Choice value="sadness" />
    <Choice value="surprise" />
    <Choice value="neutral" />
  </Choices>

  <Choices name="empathy" toName="dialogue" choice="single" showInLine="true">
    <Choice value="2" alias="Highly Empathetic (+2)" />
    <Choice value="1" alias="Supportive (+1)" />
    <Choice value="0" alias="Neutral (0)" />
    <Choice value="-1" alias="Low Empathy (-1)" />
    <Choice value="-2" alias="Hostile (-2)" />
  </Choices>

  <Choices name="uncertain" toName="dialogue" choice="single" showInLine="true">
    <Choice value="false" alias="Confident" />
    <Choice value="true" alias="Uncertain" />
  </Choices>
</View>
```

### **Manual JSONL Editing (Alternative)**
Edit `data/labeled/dialogues.jsonl` directly in VS Code. Use JSONL syntax highlighting.

---

## 📈 6. Quality Targets (Milestone 1)

| Metric | Target |
|--------|--------|
| **Total gold lines** | ≥ 2,000 |
| **Per-emotion minimum** | ≥ 150 lines each |
| **Per-empathy minimum** | ≥ 200 lines each |
| **Uncertain rate** | < 15% |
| **Inter-annotator agreement** | Cohen's κ > 0.7 (if 2+ annotators) |
| **Weak-label correction rate** | 100% of silver labels reviewed |

**Recommended splits** (after shuffling):
- Train: 70% (1,400+)
- Val: 15% (300+)
- Test: 15% (300+)

---

## 🚀 7. Silver Label Generation (Weak Supervision)

If starting from zero labels:

```bash
# 1. Run current pipeline to generate weak labels
streamlit run app.py
# Upload each screenplay → Run pipeline → Export CSV

# 2. Convert to JSONL with source=weak_label
python scripts/weak_to_jsonl.py \
  --input data/processed/dialogues_final_scored.csv \
  --output data/labeled/silver_dialogues.jsonl

# 3. Import silver_dialogues.jsonl into Label Studio
# 4. Correct each line (change emotion/empathy/uncertain as needed)
# 5. Export as gold_dialogues.jsonl
```

> **Critical**: Every weak label MUST be human-reviewed. Do not train on uncorrected silver labels.

---

## 📞 8. Validation Checklist

Before training, run `python -m data.dataset_audit` and verify:

- [ ] All required fields present
- [ ] No invalid emotion values
- [ ] No invalid empathy values
- [ ] Class balance acceptable (no class < 100)
- [ ] Uncertain rate < 15%
- [ ] Train/val/test splits created
- [ ] No duplicate `dialogue`+`character`+`movie` combinations

---

*Milestone 1 Version — Compact Schema — 2025-11-21*
