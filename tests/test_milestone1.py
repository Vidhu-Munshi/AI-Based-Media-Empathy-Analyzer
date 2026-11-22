import json
from data.dataset_audit import audit_dataset
from model.inference import extract_lexical_features, parse_screenplay_text

def test_parser_extracts_scene_character_and_dialogue():
    screenplay = "INT. OFFICE - DAY\n\nMAYA\nHow are you?\n\nJON\nI am fine.\n"
    rows = parse_screenplay_text(screenplay, "sample")
    assert rows.to_dict("records") == [{"Movie": "sample", "Scene": "INT. OFFICE - DAY", "Character": "MAYA", "Dialogue": "how are you?"}, {"Movie": "sample", "Scene": "INT. OFFICE - DAY", "Character": "JON", "Dialogue": "i am fine."}]

def test_lexical_features_are_deterministic():
    assert extract_lexical_features("Wait... are you sure?") == {"word_count": 4, "lexical_diversity_ttr": 1.0, "hesitation_markers": 1, "hesitation_rate_per_100_words": 25.0, "is_question": 1}

def test_dataset_audit_accepts_compact_schema(tmp_path):
    record = {"dialogue": "i hear you", "character": "MAYA", "movie": "sample", "scene": "INT. OFFICE - DAY", "emotion": "joy", "empathy": 2, "uncertain": False, "source": "gold"}
    dataset = tmp_path / "dialogues.jsonl"
    dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
    report, errors = audit_dataset(dataset)
    assert errors == []
    assert report["valid_records"] == 1
