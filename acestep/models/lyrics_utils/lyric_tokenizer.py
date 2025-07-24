import os
import re
import textwrap
from functools import cached_property

import torch
from num2words import num2words
from spacy.lang.en import English
from tokenizers import Tokenizer

from .hi_num2words import TextNorm as hi_num2words
from typing import Dict, List, Optional, Set, Union


def get_spacy_lang():
    # For Hindi/Hinglish, English tokenization works well
    return English()


def split_sentence(text, text_split_length=250):
    """Preprocess the input text"""
    text_splits = []
    if text_split_length is not None and len(text) >= text_split_length:
        text_splits.append("")
        nlp = get_spacy_lang()
        nlp.add_pipe("sentencizer")
        doc = nlp(text)
        for sentence in doc.sents:
            if len(text_splits[-1]) + len(str(sentence)) <= text_split_length:
                text_splits[-1] += " " + str(sentence)
                text_splits[-1] = text_splits[-1].lstrip()
            elif len(str(sentence)) > text_split_length:
                for line in textwrap.wrap(
                    str(sentence),
                    width=text_split_length,
                    drop_whitespace=True,
                    break_on_hyphens=False,
                    tabsize=1,
                ):
                    text_splits.append(str(line))
            else:
                text_splits.append(str(sentence))

        if len(text_splits) > 1:
            if text_splits[0] == "":
                del text_splits[0]
    else:
        text_splits = [text.lstrip()]

    return text_splits


_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = {
    "hi": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dr", "doctor"),
            ("mr", "mister"),
            ("mrs", "missus"),
            ("prof", "professor"),
            ("govt", "government"),
            ("pvt", "private"),
            ("sr", "senior"),
            ("jr", "junior"),
            ("dept", "department"),
            ("asst", "assistant"),
            ("eng", "engineer"),
            ("min", "minute"),
            ("sec", "second"),
            ("hrs", "hours"),
        ]
    ]
}


def expand_abbreviations_multilingual(text, lang="hi"):
    for regex, replacement in _abbreviations[lang]:
        text = re.sub(regex, replacement, text)
    return text


_symbols_multilingual = {
    "hi": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " and "),
            ("@", " at "),
            ("%", " percent "),
            ("#", " number "),
            ("$", " dollar "),
            ("₹", " rupees "),
            ("£", " pound "),
            ("°", " degree "),
            ("+", " plus "),
            ("=", " equal "),
            ("~", " approximately "),
        ]
    ]
}


def expand_symbols_multilingual(text, lang="hi"):
    for regex, replacement in _symbols_multilingual[lang]:
        text = re.sub(regex, replacement, text)
        text = text.replace("  ", " ")  # Ensure there are no double spaces
    return text.strip()


_ordinal_re = {
    "hi": re.compile(r"([0-9]+)(st|nd|rd|th|va|wa)")  # Handle both English and Hinglish ordinal suffixes
}
_number_re = re.compile(r"[0-9]+")
_currency_re = {
    "USD": re.compile(r"((\$[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+\$))"),
    "GBP": re.compile(r"((£[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+£))"),
    "EUR": re.compile(r"(([0-9\.\,]*[0-9]+€)|((€[0-9\.\,]*[0-9]+)))"),
    "INR": re.compile(r"((₹[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+₹))")
}

_comma_number_re = re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b")
_decimal_number_re = re.compile(r"([0-9]+[.,][0-9]+)")


def _remove_commas(m):
    text = m.group(0)
    if "," in text:
        text = text.replace(",", "")
    return text


def _expand_decimal_point(m):
    amount = m.group(1).replace(",", ".")
    return num2words(float(amount), lang="en")


def _expand_currency(m, currency="INR"):
    amount = float((re.sub(r"[^\d.]", "", m.group(0).replace(",", "."))))
    full_amount = num2words(amount, to="currency", currency=currency, lang="en")
    if amount.is_integer():
        last_and = full_amount.rfind(", ")
        if last_and != -1:
            full_amount = full_amount[:last_and]
    return full_amount


def _expand_ordinal(m):
    return num2words(int(m.group(1)), ordinal=True, lang="en")


def _expand_number(m):
    return num2words(int(m.group(0)), lang="en")


def expand_numbers_multilingual(text):
    text = hi_num2words()(text)
    text = re.sub(_comma_number_re, _remove_commas, text)
    try:
        text = re.sub(_currency_re["INR"], lambda m: _expand_currency(m, "INR"), text)
        text = re.sub(_currency_re["USD"], lambda m: _expand_currency(m, "USD"), text)
        text = re.sub(_currency_re["GBP"], lambda m: _expand_currency(m, "GBP"), text)
        text = re.sub(_currency_re["EUR"], lambda m: _expand_currency(m, "EUR"), text)
    except:
        pass
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re["hi"], _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def multilingual_cleaners(text):
    text = text.replace('"', "")
    text = lowercase(text)
    try:
        text = expand_numbers_multilingual(text)
    except:
        pass
    try:
        text = expand_abbreviations_multilingual(text)
    except:
        pass
    try:
        text = expand_symbols_multilingual(text)
    except:
        pass
    text = collapse_whitespace(text)
    return text


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


DEFAULT_VOCAB_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "vocab.json"
)


class LyricTokenizer:
    def __init__(self, vocab_file=DEFAULT_VOCAB_FILE):
        self.tokenizer = None
        if vocab_file is not None:
            self.tokenizer = Tokenizer.from_file(vocab_file)
        self.char_limit = 2000  # Increased limit for Hinglish to handle code-mixing

    def check_input_length(self, txt):
        if len(txt) > self.char_limit:
            print(
                f"[!] Warning: The text length exceeds the character limit of {self.char_limit}, this might cause truncated audio."
            )

    def preprocess_text(self, txt):
        # Special handling for Hinglish
        txt = hinglish_cleaners(txt)
        # Handle code-mixed text
        txt = self._handle_code_mixed(txt)
        # Apply general multilingual cleaning
        txt = multilingual_cleaners(txt)
        return txt

    def _handle_code_mixed(self, text):
        """Handle code-mixed Hinglish text"""
        # Split into words
        words = text.split()
        processed_words = []
        
        for word in words:
            # Check if word contains any Devanagari characters
            if re.search(r'[\u0900-\u097F]', word):
                # Handle Hindi word
                processed_words.append(word)
            else:
                # Handle potential Hinglish/English word
                word = self._normalize_hinglish_word(word)
                processed_words.append(word)
        
        return ' '.join(processed_words)

    def _normalize_hinglish_word(self, word):
        """Normalize Hinglish words to standard form"""
        # Common Hinglish word variations
        variations = {
            'hai': ['he', 'hein', 'hey'],
            'nahi': ['nhi', 'nahin', 'nay', 'naa'],
            'kya': ['kia', 'kiya'],
            'aur': ['or', 'arr'],
            'bahut': ['bohot', 'bhot', 'bohut'],
            'pyaar': ['pyar', 'piyar'],
            'karna': ['krna'],
            'hona': ['hna'],
            'jana': ['jna'],
            'dena': ['dna'],
            'lena': ['lna']
        }
        
        # Check if word matches any variation
        lower_word = word.lower()
        for standard, vars in variations.items():
            if lower_word in vars:
                return standard
        
        return word

    def encode(self, txt):
        self.check_input_length(txt)
        txt = self.preprocess_text(txt)
        txt = f"[hi]{txt}"
        txt = txt.replace(" ", "[SPACE]")
        return self.tokenizer.encode(txt).ids

    def decode(self, seq, skip_special_tokens=False):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(" ", "")
        txt = txt.replace("[SPACE]", " ")
        txt = txt.replace("[STOP]", "")
        return txt

    def batch_decode(
        self,
        sequences: Union[
            List[int], List[List[int]], "np.ndarray", "torch.Tensor", "tf.Tensor"
        ],
        skip_special_tokens: bool = False,
    ) -> List[str]:
        return [self.decode(seq) for seq in sequences]

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def get_number_tokens(self):
        return max(self.tokenizer.get_vocab().values()) + 1


def hinglish_cleaners(text):
    """Enhanced Hinglish-specific text cleaning"""
    # Common Hinglish contractions and variations
    hinglish_contractions = {
        # Numbers and numerals
        r'(\d+)st': r'\1first',
        r'(\d+)nd': r'\1second',
        r'(\d+)rd': r'\1third',
        r'(\d+)th': r'\1th',
        
        # Common Hinglish variations
        r'\b(hai|he|hein)\b': 'hai',
        r'\b(me|mei|mein)\b': 'mein',
        r'\b(ko|ku|koo)\b': 'ko',
        r'\b(aur|or|arr)\b': 'aur',
        r'\b(nahi|nhi|nahin|nay|naa)\b': 'nahi',
        r'\b(kya|kia|kiya)\b': 'kya',
        r'\b(bohot|bhot|bahut|bohut)\b': 'bahut',
        r'\b(pyar|pyaar|piyar)\b': 'pyaar',
        
        # Common verb forms
        r'\b(karna|krna)\b': 'karna',
        r'\b(hona|hna)\b': 'hona',
        r'\b(jana|jna)\b': 'jana',
        r'\b(dena|dna)\b': 'dena',
        r'\b(lena|lna)\b': 'lena',
        
        # Tense markers
        r'\b(raha|rha)\b': 'raha',
        r'\b(rahi|rhi)\b': 'rahi',
        r'\b(rahe|rhe)\b': 'rahe',
        
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
        
        # Common suffixes
        r'(\w+)wa\b': r'\1',  # Remove diminutive suffix
        r'(\w+)ji\b': r'\1',  # Remove respectful suffix
        
        # Handle repeated characters (common in informal writing)
        r'(.)\1{2,}': r'\1\1'  # Convert 'heyyy' to 'heyy'
    }
    
    # Apply all contractions
    for pattern, replacement in hinglish_contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Handle mixed English-Hindi numerals
    text = re.sub(r'\b1st\b', 'first', text)
    text = re.sub(r'\b2nd\b', 'second', text)
    text = re.sub(r'\b3rd\b', 'third', text)
    text = re.sub(r'\b(\d+)th\b', r'\1th', text)
    
    return text
