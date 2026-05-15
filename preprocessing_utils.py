"""
Text preprocessing utilities for the ML pipeline
Reusable components for text analytics
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

def ensure_nltk_resource(resource_paths, package_name):
    """Download an NLTK package only when none of its expected paths exist."""
    for resource_path in resource_paths:
        try:
            nltk.data.find(resource_path)
            return
        except LookupError:
            continue
    nltk.download(package_name)


# Download NLTK resources when they are missing.
ensure_nltk_resource(('tokenizers/punkt', 'tokenizers/punkt.zip'), 'punkt')
ensure_nltk_resource(('tokenizers/punkt_tab', 'tokenizers/punkt_tab.zip'), 'punkt_tab')
ensure_nltk_resource(('corpora/stopwords', 'corpora/stopwords.zip'), 'stopwords')
ensure_nltk_resource(('corpora/wordnet', 'corpora/wordnet.zip'), 'wordnet')

# Initialize preprocessing tools
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

RATING_PHRASE_EXPANSIONS = (
    (r'\b(?:5|five)\s*(?:/|out\s+of)\s*(?:5|five)(?:[\s-]*stars?)?\b', 'excellent amazing'),
    (r'\b(?:4|four)\s*(?:/|out\s+of)\s*(?:5|five)(?:[\s-]*stars?)?\b', 'good enjoyable'),
    (r'\b(?:3|three)\s*(?:/|out\s+of)\s*(?:5|five)(?:[\s-]*stars?)?\b', 'average okay'),
    (r'\b(?:2|two)\s*(?:/|out\s+of)\s*(?:5|five)(?:[\s-]*stars?)?\b', 'bad disappointing'),
    (r'\b(?:1|one|0|zero)\s*(?:/|out\s+of)\s*(?:5|five)(?:[\s-]*stars?)?\b', 'terrible awful'),
    (r'\b(?:5|five)[\s-]*stars?\b', 'excellent amazing'),
    (r'\b(?:4|four)[\s-]*stars?\b', 'good enjoyable'),
    (r'\b(?:3|three)[\s-]*stars?\b', 'average okay'),
    (r'\b(?:2|two)[\s-]*stars?\b', 'bad disappointing'),
    (r'\b(?:1|one|0|zero)[\s-]*stars?\b', 'terrible awful'),
)

class TextPreprocessor:
    """
    Complete text preprocessing pipeline for IMDB reviews
    Covers Unit I: Text Preprocessing from CSI324 course
    """
    
    @staticmethod
    def lowercase(text):
        """Convert text to lowercase"""
        return text.lower()

    @staticmethod
    def expand_rating_phrases(text):
        """Convert common star-rating phrases into sentiment-bearing words."""
        for pattern, replacement in RATING_PHRASE_EXPANSIONS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    @staticmethod
    def remove_html(text):
        """Remove HTML tags if present"""
        html_pattern = re.compile(r'<.*?>')
        return html_pattern.sub(r'', text)
    
    @staticmethod
    def tokenize(text):
        """
        Tokenization: Break text into individual tokens
        CO2: Demonstrate tokenization techniques
        """
        tokens = word_tokenize(text)
        return tokens
    
    @staticmethod
    def remove_stopwords(tokens):
        """
        Remove stopwords from token list
        CO2: Demonstrate stopword removal
        """
        filtered_tokens = [token for token in tokens 
                          if token.isalpha() and token not in STOP_WORDS]
        return filtered_tokens
    
    @staticmethod
    def apply_stemming(tokens):
        """
        Porter Stemming: Reduce words to root form
        CO2: Demonstrate stemming techniques
        """
        stemmed_tokens = [STEMMER.stem(token) for token in tokens]
        return stemmed_tokens
    
    @staticmethod
    def apply_lemmatization(tokens):
        """
        Lemmatization: Convert words to base/dictionary form
        CO2: Demonstrate lemmatization techniques
        """
        lemmatized_tokens = [LEMMATIZER.lemmatize(token) for token in tokens]
        return lemmatized_tokens
    
    @staticmethod
    def preprocess_basic(text):
        """
        Basic preprocessing: Lowercase, Tokenize, Remove Stopwords
        Suitable for TF-IDF and BoW approaches
        """
        # Step 1: Normalize common rating phrases, then lowercase
        text = TextPreprocessor.expand_rating_phrases(text)
        text = TextPreprocessor.lowercase(text)
        
        # Step 2: Remove HTML (if present)
        text = TextPreprocessor.remove_html(text)
        
        # Step 3: Tokenization
        tokens = TextPreprocessor.tokenize(text)
        
        # Step 4: Remove Stopwords
        tokens = TextPreprocessor.remove_stopwords(tokens)
        
        return tokens
    
    @staticmethod
    def preprocess_advanced(text):
        """
        Advanced preprocessing: Lowercase, Tokenize, Remove Stopwords, Lemmatization
        Provides better feature quality for classification models
        CO1, CO2, CO3: Covers fundamental preprocessing and feature engineering
        """
        # Step 1: Basic preprocessing
        tokens = TextPreprocessor.preprocess_basic(text)
        
        # Step 2: Lemmatization
        tokens = TextPreprocessor.apply_lemmatization(tokens)
        
        return tokens
    
    @staticmethod
    def preprocess_text(text, use_lemmatization=True):
        """
        Complete pipeline - wrapper function for flexibility
        
        Parameters:
        -----------
        text : str
            Input text to preprocess
        use_lemmatization : bool
            Whether to apply lemmatization (default: True)
        
        Returns:
        --------
        str
            Preprocessed text as space-joined string
        """
        if use_lemmatization:
            tokens = TextPreprocessor.preprocess_advanced(text)
        else:
            tokens = TextPreprocessor.preprocess_basic(text)
        
        return ' '.join(tokens)


class TextStatistics:
    """Calculate statistics on text data"""
    
    @staticmethod
    def get_text_length(text):
        """Return length of text"""
        return len(text)
    
    @staticmethod
    def get_word_count(text):
        """Return number of words"""
        return len(text.split())
    
    @staticmethod
    def get_unique_words(text):
        """Return number of unique words"""
        words = text.lower().split()
        return len(set(words))
    
    @staticmethod
    def get_average_word_length(text):
        """Return average word length"""
        words = text.split()
        if not words:
            return 0
        return sum(len(word) for word in words) / len(words)


if __name__ == "__main__":
    # Test the preprocessor
    sample_text = "This is an excellent movie! I really enjoyed watching it."
    
    print("Original text:")
    print(sample_text)
    print()
    
    print("Basic preprocessing:")
    basic_result = TextPreprocessor.preprocess_text(sample_text, use_lemmatization=False)
    print(basic_result)
    print()
    
    print("Advanced preprocessing (with lemmatization):")
    advanced_result = TextPreprocessor.preprocess_text(sample_text, use_lemmatization=True)
    print(advanced_result)
    print()
    
    print("Text Statistics:")
    print(f"Original length: {TextStatistics.get_text_length(sample_text)}")
    print(f"Word count: {TextStatistics.get_word_count(sample_text)}")
    print(f"Unique words: {TextStatistics.get_unique_words(sample_text)}")
    print(f"Average word length: {TextStatistics.get_average_word_length(sample_text):.2f}")
