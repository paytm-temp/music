import re
import string

EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # Emoticons
    "]+",
    flags=re.UNICODE,
)

# Translation table for character replacements
TRANSLATION_TABLE = str.maketrans(
    {
        "-": " ",  # Replace '-' with space
        ",": None,
        ".": None,
        "，": None,
        "。": None,
        "!": None,
        "！": None,
        "?": None,
        "？": None,
        "…": None,
        ";": None,
        "；": None,
        ":": None,
        "：": None,
        "\u3000": " ",  # Replace full-width space with regular space
    }
)

# Pattern for content in parentheses
BACKSLASH_PATTERN = re.compile(r"\(.*?\)|\[.*?\]")

# Pattern for multiple spaces
SPACE_PATTERN = re.compile("(?<!^)\s+(?!$)")

# Hinglish specific patterns
HINGLISH_PATTERNS = {
    # Common variations
    r'\b(hai|he|hein)\b': 'hai',
    r'\b(me|mei|mein)\b': 'mein',
    r'\b(ko|ku|koo)\b': 'ko',
    r'\b(aur|or|arr)\b': 'aur',
    r'\b(nahi|nhi|nahin|nay|naa)\b': 'nahi',
    r'\b(kya|kia|kiya)\b': 'kya',
    r'\b(bohot|bhot|bahut|bohut)\b': 'bahut',
    r'\b(pyar|pyaar|piyar)\b': 'pyaar',
    # Common endings
    r'\b(wala|vala|walla)\b': 'wala',
    r'\b(kar|kr)\b': 'kar',
    # Common words
    r'\b(acha|accha|achchha)\b': 'accha',
    r'\b(thik|theek|teek|thek)\b': 'theek',
    # Chat style writing
    r'\b(plz|pls|plij)\b': 'please',
    r'\b(u)\b': 'you',
    r'\b(nd|n)\b': 'and',
}

def normalize_hinglish_text(text):
    """
    Normalize Hinglish text by standardizing common variations
    """
    # Convert to lowercase
    text = text.lower()
    
    # Apply Hinglish patterns
    for pattern, replacement in HINGLISH_PATTERNS.items():
        text = re.sub(pattern, replacement, text)
    
    # Handle repeated characters (e.g., "helloo" -> "hello")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Handle common number substitutions
    text = re.sub(r'\b2\b', 'to', text)
    text = re.sub(r'\b4\b', 'for', text)
    text = re.sub(r'\b(gr8|gr8t)\b', 'great', text)
    
    return text

class LyricNormalizer:
    def normalize(self, text):
        """
        Normalize text with Hindi/Hinglish-specific handling
        """
        # Step 1: Replace '-' with ' ' and remove punctuation
        text = text.translate(TRANSLATION_TABLE)

        # Step 2: Remove emoji
        text = EMOJI_PATTERN.sub("", text)

        # Step 3: Replace consecutive whitespace with single space
        text = SPACE_PATTERN.sub(" ", text)

        # Step 4: Strip whitespace
        text = text.strip()

        # Step 5: Apply Hinglish normalization
        text = normalize_hinglish_text(text)
        
        return text
