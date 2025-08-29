# rag/utils.py

import yaml
import re
from typing import List
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# --- Config Helper ---
import yaml

# --- Config Helper ---
def load_and_format_config(config_path: str, context: dict | None = None) -> dict:
    """
    Loads a YAML config file and formats string values using template variables.

    Args:
        config_path (str): The path to the YAML configuration file.
        context (dict | None, optional): An external dictionary of template variables.
                                         If None, uses the config file itself for context.
                                         Defaults to None.

    Returns:
        dict: The loaded and formatted configuration dictionary.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    def _format_values(obj, context):
        if isinstance(obj, dict): return {k: _format_values(v, context) for k, v in obj.items()}
        if isinstance(obj, list): return [_format_values(elem, context) for elem in obj]
        if isinstance(obj, str):
            try:
                return obj.format(**context)
            except KeyError:
                return obj
        return obj

    # Use the provided external context if it exists, otherwise default to the old behavior.
    formatting_context = context if context is not None else config

    return _format_values(config, formatting_context)


# --- PHRASE TOKENIZATION HELPERS ---

SQUASH_PHRASES = [
    "volley straight drive", "volley deep drive", "volley hard drive",
    "cross down the middle", "volley cross-court nick", "volley 2-wall boast",
    "volley 3-wall boast", "volley reverse boast", "volley straight lob",
    "volley cross lob", "volley cross kill", "volley straight kill",
    "volley hard cross", "deep drive", "hard drive", "straight drive",
    "volley drive", "cross-court", "cross court", "cross lob", "lob cross",
    "deep cross", "cross deep", "cross wide", "wide cross", "hard cross",
    "volley cross", "cross-court nick", "counter drop", "cross drop",
    "straight drop", "volley cross drop", "volley straight drop",
    "2-wall boast", "3-wall boast", "trickle boast", "reverse boast",
    "straight lob", "volley drop", "volley lob", "volley flick"
]


def replace_phrases(text: str, phrases: List[str], debug=False) -> str:
    """
    Replaces multi-word phrases in text with a single underscored token.
    Processes longest phrases first and handles simple pluralization ('s').
    """
    sorted_phrases = sorted(phrases, key=len, reverse=True)
    processed_text = text

    if debug:
        print(f"[PHRASE DEBUG] Input text: '{text}'")

    for phrase in sorted_phrases:
        if phrase.lower() in text.lower():  # Quick check if phrase might be present
            underscored_phrase = phrase.replace(" ", "_").replace("-", "_")

            # Create regex pattern for the exact phrase
            escaped_phrase = re.escape(phrase)

            # Pattern 1: Exact phrase match
            pattern1 = r'\b' + escaped_phrase + r'\b'

            # Pattern 2: Phrase with 's' added to the end
            pattern2 = r'\b' + escaped_phrase + r's\b'

            # Count matches before replacement
            matches1 = len(re.findall(pattern1, processed_text, flags=re.IGNORECASE))
            matches2 = len(re.findall(pattern2, processed_text, flags=re.IGNORECASE))

            if debug and (matches1 > 0 or matches2 > 0):
                print(f"[PHRASE DEBUG] Found phrase '{phrase}': {matches1} exact matches, {matches2} plural matches")

            # Apply both patterns
            processed_text = re.sub(pattern1, underscored_phrase, processed_text, flags=re.IGNORECASE)
            processed_text = re.sub(pattern2, underscored_phrase, processed_text, flags=re.IGNORECASE)

    if debug:
        print(f"[PHRASE DEBUG] Final processed text: '{processed_text}'")

    return processed_text


# --- FINAL TOKENIZER ---

def advanced_tokenizer(text: str) -> List[str]:
    """
    Final tokenizer: Replaces phrases, tokenizes (preserving hyphens), stems, and removes stop words.
    """
    stemmer = PorterStemmer()

    # Using the "minimal" stop word list for the meta-index strategy
    standard_stop_words = set(stopwords.words('english'))
    stemmed_stop_words = {stemmer.stem(w) for w in standard_stop_words}

    # 1. Pre-process the text to replace known phrases first
    phrase_replaced_text = replace_phrases(text, SQUASH_PHRASES)

    # 2. Tokenize the processed text
    tokens = re.findall(r'\b\w[\w-]*\b', phrase_replaced_text.lower())

    # 3. Stem the tokens
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # 4. Filter stop words
    filtered_tokens = [token for token in stemmed_tokens if token not in stemmed_stop_words]

    return filtered_tokens