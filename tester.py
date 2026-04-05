import os
import json


def _load_entries():
    if not os.path.exists("data/tester_data.json"):
        return []
    try:
        with open("data/tester_data.json", "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            data = json.loads(content)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_entries(entries):
    with open("data/tester_data.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

def add_response(input_text, response1, response2):
    text = _load_entries()

    text.append({
        "input": input_text,
        "response1": response1,
        "response2": response2,
        "choice": None
    })

    _save_entries(text)

def validate_response(response1, selected_response):
    text = _load_entries()

    for entry in text:
        if entry.get("response1") == response1 and entry.get("choice") is None:
            entry["choice"] = selected_response
            break

    _save_entries(text)