# data_utils.py
import json
import os

# Đường dẫn tĩnh đến file user_data.json
DB_PATH = "/src/user_data.json"


def load_json(path=DB_PATH):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_json(data, path=DB_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)