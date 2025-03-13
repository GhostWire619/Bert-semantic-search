import re
import unicodedata
from typing import List

def preprocess_text(text: str) -> str:
    """
    Preprocess text by performing the following operations:
    1. Convert to lowercase
    2. Remove special characters and numbers
    3. Remove extra whitespace
    4. Normalize unicode characters
    """
    # Convert to lowercase
    text = text.lower()
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words.
    """
    return text.split()

def remove_stopwords(text: str, stopwords: List[str] = None) -> str:
    """
    Remove stopwords from text.
    If no stopwords list is provided, returns the original text.
    """
    if not stopwords:
        return text
        
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered_words) 