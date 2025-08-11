"""
Text preprocessing utilities for benchmark detection.

This module provides text cleaning, normalization, and preprocessing
functions to improve detection accuracy.
"""

import re
import string
import logging
from typing import List, Optional
import unicodedata


class TextPreprocessor:
    """
    Text preprocessing utility for cleaning and normalizing input text.
    
    Provides various text cleaning operations to standardize text
    before similarity matching and detection.
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        remove_extra_whitespace: bool = True,
        normalize_unicode: bool = True,
        min_length: int = 3
    ):
        """
        Initialize the text preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            remove_extra_whitespace: Normalize whitespace
            normalize_unicode: Normalize Unicode characters
            min_length: Minimum text length to process
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_unicode = normalize_unicode
        self.min_length = min_length
        
        self.logger = logging.getLogger(__name__)
        
        # Precompile regex patterns for efficiency
        self.whitespace_pattern = re.compile(r'\s+')
        self.punctuation_pattern = re.compile(f'[{re.escape(string.punctuation)}]')
        self.number_pattern = re.compile(r'\d+')
        
        # Common patterns to normalize
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
    
    def clean_text(self, text: str) -> str:
        """
        Apply all enabled cleaning operations to text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned and normalized text
        """
        if not text or len(text.strip()) < self.min_length:
            return ""
        
        # Start with the original text
        cleaned = text
        
        # Unicode normalization
        if self.normalize_unicode:
            cleaned = self._normalize_unicode(cleaned)
        
        # Remove URLs, emails, mentions, hashtags
        cleaned = self._remove_social_patterns(cleaned)
        
        # Convert to lowercase
        if self.lowercase:
            cleaned = cleaned.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            cleaned = self._remove_punctuation(cleaned)
        
        # Normalize whitespace
        if self.remove_extra_whitespace:
            cleaned = self._normalize_whitespace(cleaned)
        
        return cleaned.strip()
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        try:
            # Normalize to NFC form (canonical decomposition + canonical composition)
            normalized = unicodedata.normalize('NFC', text)
            
            # Remove non-printable characters
            printable = ''.join(char for char in normalized if unicodedata.category(char)[0] != 'C')
            
            return printable
        except Exception as e:
            self.logger.warning(f"Unicode normalization failed: {str(e)}")
            return text
    
    def _remove_social_patterns(self, text: str) -> str:
        """Remove URLs, emails, mentions, and hashtags."""
        # Replace with spaces to maintain word boundaries
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self.mention_pattern.sub(' ', text)
        text = self.hashtag_pattern.sub(' ', text)
        return text
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation marks."""
        return self.punctuation_pattern.sub(' ', text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace (multiple spaces, tabs, newlines to single space)."""
        return self.whitespace_pattern.sub(' ', text)
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract important keywords from text.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        # Clean the text
        cleaned = self.clean_text(text)
        
        if not cleaned:
            return []
        
        # Split into words
        words = cleaned.split()
        
        # Filter out very short words and common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        keywords = []
        for word in words:
            if (len(word) >= 3 and 
                word.lower() not in stop_words and 
                not word.isdigit()):
                keywords.append(word)
        
        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for keyword in keywords:
            if keyword.lower() not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword.lower())
        
        return unique_keywords[:max_keywords]
    
    def get_text_features(self, text: str) -> dict:
        """
        Extract various text features for analysis.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of text features
        """
        cleaned = self.clean_text(text)
        
        features = {
            'original_length': len(text),
            'cleaned_length': len(cleaned),
            'word_count': len(cleaned.split()) if cleaned else 0,
            'char_count': len(cleaned),
            'avg_word_length': 0,
            'has_numbers': bool(self.number_pattern.search(text)),
            'has_punctuation': any(char in string.punctuation for char in text),
            'has_uppercase': any(char.isupper() for char in text),
            'keywords': self.extract_keywords(text),
        }
        
        # Calculate average word length
        words = cleaned.split()
        if words:
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)
        
        return features
