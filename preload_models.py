"""
preload_models.py
Preloads MarianMT models into cache at startup to reduce first-call latency.
"""
import sys, os
sys.path.insert(0, os.path.abspath("src"))

from health_translator.translator import load_model_and_tokenizer, get_marian_name
import joblib
import os
import sentencepiece as spm 

# languages you want to preload (you can modify)
LANG_PAIRS = [
    ("en", "hi"), ("hi", "en"),
    ("en", "ta"), ("ta", "en"),
    ("en", "te"), ("te", "en"),
    ("en", "ml"), ("ml", "en"),
]

def preload():
    for src, tgt in LANG_PAIRS:
        model_name = get_marian_name(src, tgt)
        try:
            load_model_and_tokenizer(model_name)
            print(f"✅ Preloaded {model_name}")
        except Exception as e:
            print(f"⚠️ Could not load {model_name}: {e}")

if __name__ == "__main__":
    preload()
