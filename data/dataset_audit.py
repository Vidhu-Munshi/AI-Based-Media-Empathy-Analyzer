"""Validate and summarize a Milestone 1 dialogue-label JSONL dataset.

Run with: python -m data.dataset_audit --input data/labeled/dialogues.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

EMOTIONS = {"anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"}
EMPATHY_SCORES = {-2, -1, 0, 1, 2}
SOURCES = {"gold", "weak_label"}
REQUIRED_FIELDS = {"dialogue", "character", "movie", "scene", "emotion", "empathy", "uncertain", "source"}


def validate_record(record: Any, line_number: int) -> list[str]:
    """Return human-readable schema errors for one JSONL record."""
    if not isinstance(record, dict):
        return [f"line {line_number}: expected a JSON object"]
    errors = []
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        return [f"line {line_number}: missing fields: {', '.join(sorted(missing))}"]
    for field in ("dialogue", "character", "movie", "scene"):
        if not isinstance(record[field], str) or not record[field].strip():
            errors.append(f"line {line_number}: {field} must be a non-empty string")
    if record["emotion"] not in EMOTIONS:
        errors.append(f"line {line_number}: invalid emotion {record['emotion']!r}")
    if type(record["empathy"]) is not int or record["empathy"] not in EMPATHY_SCORES:
        errors.append(f"line {line_number}: empathy must be one of -2, -1, 0, 1, 2")
    if type(record["uncertain"]) is not bool:
        errors.append(f"line {line_number}: uncertain must be a boolean")
    if record["source"] not in SOURCES:
        errors.append(f"line {line_number}: source must be gold or weak_label")
    return errors


def audit_dataset(path: Path) -> tuple[dict[str, Any], list[str]]:
    """Read, validate, and summarize a JSONL label file."""
    records, errors = [], []
    if not path.is_file():
        return {}, [f"input file does not exist: {path}"]
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            errors.append(f"line {number}: blank lines are not allowed")
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {number}: invalid JSON ({exc.msg})")
            continue
        record_errors = validate_record(record, number)
        errors.extend(record_errors)
        if not record_errors:
            records.append(record)
    duplicate_keys = Counter((r["dialogue"], r["character"], r["movie"]) for r in records)
    report = {
        "valid_records": len(records), "invalid_records": len(errors),
        "emotion_distribution": dict(sorted(Counter(r["emotion"] for r in records).items())),
        "empathy_distribution": dict(sorted(Counter(r["empathy"] for r in records).items())),
        "movie_distribution": dict(sorted(Counter(r["movie"] for r in records).items())),
        "character_count": len({r["character"] for r in records}),
        "source_distribution": dict(sorted(Counter(r["source"] for r in records).items())),
        "uncertain_count": sum(r["uncertain"] for r in records),
        "duplicate_dialogue_character_movie_records": sum(count - 1 for count in duplicate_keys.values() if count > 1),
    }
    return report, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a Milestone 1 dialogue-label JSONL file.")
    parser.add_argument("--input", required=True, type=Path, help="Path to dialogues.jsonl")
    args = parser.parse_args()
    report, errors = audit_dataset(args.input)
    if report:
        print(json.dumps(report, indent=2))
    if errors:
        print("\nValidation errors:")
        print("\n".join(errors))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
