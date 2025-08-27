import json
import re


def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```[^\n]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    decoder = json.JSONDecoder()
    for i, c in enumerate(text):
        if c in "[{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("No valid JSON found")
