

"""
translator.py
- Auto-detect language (langdetect)
- Translate text between English and regional languages using MarianMT (Helsinki-NLP)
- Caches tokenizers/models to avoid re-downloading
"""

import os
from typing import Tuple
from langdetect import detect, DetectorFactory
from transformers import MarianMTModel, MarianTokenizer
import torch

DetectorFactory.seed = 0  # deterministic langdetect

# Supported language codes (add more as needed)
# two-letter iso codes used by langdetect
SUPPORTED = {"hi","ta","te","ml","kn","bn","mr","gu","pa","ur","en"}

# cache dicts
_MODEL_CACHE = {}
_TOKENIZER_CACHE = {}

def detect_language(text: str) -> str:
    """Return language code using langdetect (e.g., 'en', 'hi')."""
    text = (text or "").strip()
    if text == "":
        return "en"
    try:
        lang = detect(text)
        return lang
    except Exception:
        return "en"

def get_marian_name(src: str, tgt: str) -> str:
    """Return best candidate model name for src->tgt.
    Try direct 'opus-mt-{src}-{tgt}'. If not supported, return None (caller will fallback via en).
    """
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    return model_name

def load_model_and_tokenizer(model_name: str):
    """Load and cache Marian model + tokenizer for given model_name."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name], _TOKENIZER_CACHE[model_name]

    print(f"Loading model: {model_name} ... (this may take a moment)")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    _MODEL_CACHE[model_name] = model
    _TOKENIZER_CACHE[model_name] = tokenizer
    return model, tokenizer

def translate_with_model(model_name: str, texts):
    """Translate list of texts using the specified Marian model."""
    model, tokenizer = load_model_and_tokenizer(model_name)
    # device placement
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # tokenize
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    translated = model.generate(**batch, num_beams=4, max_length=512)
    outs = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return outs

def translate(text: str, target_lang: str = "en") -> Tuple[str,str]:
    """
    Translate text into target_lang.
    Returns (detected_source, translated_text)
    Workflow:
      - detect source lang
      - if source == target -> return text
      - try direct model src->tgt
      - else fallback: src->en (if src != 'en') then en->tgt (if tgt != 'en')
    """
    src = detect_language(text)
    src = src if src in SUPPORTED else src  # still use detected code; model availability checked later
    target = target_lang

    if src == target:
        return src, text

    # try direct model src->target
    direct_model = get_marian_name(src, target)
    try:
        if direct_model:
            # attempt to load; if not available, transformers will raise
            translated = translate_with_model(direct_model, [text])[0]
            return src, translated
    except Exception:
        # direct model not available or failed; we will fallback
        pass

    # fallback route: src -> en -> target
    try:
        if src != "en":
            model_se = get_marian_name(src, "en")
            translated_en = translate_with_model(model_se, [text])[0]
        else:
            translated_en = text

        if target != "en":
            model_et = get_marian_name("en", target)
            translated_final = translate_with_model(model_et, [translated_en])[0]
        else:
            translated_final = translated_en

        return src, translated_final
    except Exception as e:
        # if all fails return original text
        print("Translation fallback failed:", str(e))
        return src, text
