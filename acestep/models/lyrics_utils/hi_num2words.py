import re
import string
import sys
from typing import Dict, List, Optional, Set, Union

# Define number systems
HINDI_NUMBERS = {
    '0': 'zero',
    '1': 'ek',
    '2': 'do',
    '3': 'teen',
    '4': 'char',
    '5': 'paanch',
    '6': 'che',
    '7': 'saat',
    '8': 'aath',
    '9': 'nau',
    '10': 'das',
    '11': 'gyarah',
    '12': 'barah',
    '13': 'terah',
    '14': 'chaudah',
    '15': 'pandrah',
    '16': 'solah',
    '17': 'satrah',
    '18': 'atharah',
    '19': 'unnees',
    '20': 'bees',
    '21': 'ikkees',
    '22': 'baaees',
    '23': 'teyees',
    '24': 'chaubees',
    '25': 'pachees',
    '26': 'chhabbees',
    '27': 'sattaees',
    '28': 'athaees',
    '29': 'unattees',
    '30': 'tees',
    '40': 'chalees',
    '50': 'pachas',
    '60': 'saath',
    '70': 'sattar',
    '80': 'assi',
    '90': 'navve',
    '100': 'sau',
    '1000': 'hazaar',
    '100000': 'lakh',
    '10000000': 'crore'
}

# Define currency units
CURRENCY_UNITS = {
    'INR': {
        'main': 'rupaye',
        'decimal': 'paise',
        'symbol': '₹'
    },
    'USD': {
        'main': 'dollar',
        'decimal': 'cents',
        'symbol': '$'
    },
    'GBP': {
        'main': 'pound',
        'decimal': 'pence',
        'symbol': '£'
    }
}

# Define ordinal suffixes
ORDINAL_SUFFIXES = {
    '1': 'pehla',
    '2': 'doosra',
    '3': 'teesra',
    '4': 'chautha',
    '5': 'paanchva',
    '6': 'chatha',
    '7': 'saatva',
    '8': 'aathva',
    '9': 'nauva',
    '10': 'dasva'
}

# Regex patterns
NUMBER_PATTERN = re.compile(r'\d+')
DECIMAL_PATTERN = re.compile(r'\d*\.\d+')
CURRENCY_PATTERN = re.compile(r'[₹$£]\s*\d+(:?\.\d{2})?')
ORDINAL_PATTERN = re.compile(r'\d+(st|nd|rd|th)')

class TextNorm:
    def __init__(
        self,
        to_latin: bool = False,
        to_upper: bool = False,
        to_lower: bool = False,
        remove_space: bool = False,
        check_chars: bool = True,
    ):
        self.to_latin = to_latin
        self.to_upper = to_upper
        self.to_lower = to_lower
        self.remove_space = remove_space
        self.check_chars = check_chars

    def normalize_number(self, number_str: str) -> str:
        """Convert a number to its Hinglish word representation"""
        num = int(number_str)
        
        if num == 0:
            return HINDI_NUMBERS['0']
            
        if str(num) in HINDI_NUMBERS:
            return HINDI_NUMBERS[str(num)]
            
        words = []
        
        # Handle crores
        crores = num // 10000000
        if crores > 0:
            if str(crores) in HINDI_NUMBERS:
                words.append(f"{HINDI_NUMBERS[str(crores)]} crore")
            else:
                words.append(f"{self.normalize_number(str(crores))} crore")
            num = num % 10000000
            
        # Handle lakhs
        lakhs = num // 100000
        if lakhs > 0:
            if str(lakhs) in HINDI_NUMBERS:
                words.append(f"{HINDI_NUMBERS[str(lakhs)]} lakh")
            else:
                words.append(f"{self.normalize_number(str(lakhs))} lakh")
            num = num % 100000
            
        # Handle thousands
        thousands = num // 1000
        if thousands > 0:
            if str(thousands) in HINDI_NUMBERS:
                words.append(f"{HINDI_NUMBERS[str(thousands)]} hazaar")
            else:
                words.append(f"{self.normalize_number(str(thousands))} hazaar")
            num = num % 1000
            
        # Handle hundreds
        hundreds = num // 100
        if hundreds > 0:
            words.append(f"{HINDI_NUMBERS[str(hundreds)]} sau")
            num = num % 100
            
        # Handle remaining two digits
        if num > 0:
            if str(num) in HINDI_NUMBERS:
                words.append(HINDI_NUMBERS[str(num)])
            else:
                tens = (num // 10) * 10
                ones = num % 10
                if tens > 0:
                    words.append(HINDI_NUMBERS[str(tens)])
                if ones > 0:
                    words.append(HINDI_NUMBERS[str(ones)])
                
        return ' '.join(words)

    def normalize_decimal(self, decimal_str: str) -> str:
        """Convert a decimal number to its Hinglish word representation"""
        parts = decimal_str.split('.')
        whole = parts[0]
        decimal = parts[1] if len(parts) > 1 else '0'
        
        result = self.normalize_number(whole)
        if decimal != '0':
            result += f" point {' '.join(HINDI_NUMBERS[d] for d in decimal)}"
            
        return result

    def normalize_currency(self, amount_str: str) -> str:
        """Convert a currency amount to its Hinglish word representation"""
        # Remove currency symbols and spaces
        for currency in CURRENCY_UNITS.values():
            amount_str = amount_str.replace(currency['symbol'], '').strip()
            
        parts = amount_str.split('.')
        main_unit = parts[0]
        decimal_unit = parts[1] if len(parts) > 1 else '0'
        
        # Default to INR if no currency specified
        currency = CURRENCY_UNITS['INR']
        
        result = f"{self.normalize_number(main_unit)} {currency['main']}"
        if decimal_unit != '0':
            result += f" {self.normalize_number(decimal_unit)} {currency['decimal']}"
            
        return result

    def normalize_ordinal(self, ordinal_str: str) -> str:
        """Convert an ordinal number to its Hinglish word representation"""
        num = ''.join(filter(str.isdigit, ordinal_str))
        
        if num in ORDINAL_SUFFIXES:
            return ORDINAL_SUFFIXES[num]
            
        # For numbers > 10, use the number + 'va/wa'
        return f"{self.normalize_number(num)}va"

    def __call__(self, text: str) -> str:
        """Main normalization function"""
        if not text:
            return text
            
        # Handle ordinals first (1st, 2nd, etc.)
        text = ORDINAL_PATTERN.sub(
            lambda m: self.normalize_ordinal(m.group()), text
        )
        
        # Handle currency
        text = CURRENCY_PATTERN.sub(
            lambda m: self.normalize_currency(m.group()), text
        )
        
        # Handle decimals
        text = DECIMAL_PATTERN.sub(
            lambda m: self.normalize_decimal(m.group()), text
        )
        
        # Handle regular numbers
        text = NUMBER_PATTERN.sub(
            lambda m: self.normalize_number(m.group()), text
        )
        
        # Apply text transformations
        if self.to_latin:
            # Convert Devanagari to Latin script if needed
            pass
            
        if self.to_upper:
            text = text.upper()
            
        if self.to_lower:
            text = text.lower()
            
        if self.remove_space:
            text = text.replace(' ', '')
            
        return text


def test_number_conversion():
    """Test function for number conversion"""
    normalizer = TextNorm()
    test_cases = [
        ("0", "zero"),
        ("1", "ek"),
        ("10", "das"),
        ("21", "ikkees"),
        ("100", "ek sau"),
        ("1000", "ek hazaar"),
        ("100000", "ek lakh"),
        ("10000000", "ek crore"),
        ("1234567", "barah lakh chautees hazaar paanch sau satsath"),
        ("₹1234.56", "ek hazaar do sau chautees rupaye chhappan paise"),
        ("1st", "pehla"),
        ("22nd", "baaesva"),
    ]
    
    for input_text, expected in test_cases:
        result = normalizer(input_text)
        print(f"Input: {input_text}")
        print(f"Expected: {expected}")
        print(f"Got: {result}")
        print("---")


if __name__ == "__main__":
    test_number_conversion() 