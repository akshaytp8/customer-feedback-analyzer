import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    """
    Cleans and normalizes review text for AI models
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Domain-specific stopwords (app review context)
        # Why: Words like "app" appear in every review but don't add meaning
        self.custom_stopwords = {
            'app', 'application', 'update', 'version', 
            'please', 'make', 'would', 'could', 'really'
        }
        self.stop_words.update(self.custom_stopwords)
    
    def clean_text(self, text):
        """
        Deep cleaning pipeline
        
        Why each step:
        1. Lowercase: "Bug" and "bug" should be same word
        2. Remove URLs: They don't contribute to sentiment/topics
        3. Remove emojis: Can confuse NLP models (though sentiment models might use them)
        4. Remove special chars: Keep only meaningful text
        5. Remove extra spaces: Normalize formatting
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove emojis (optional - some sentiment models use them)
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        
        # Remove special characters but keep meaningful punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """
        Break text into words and normalize them
        
        Why:
        - Tokenization: "crashed" and "crashing" → separate analysis
        - Lemmatization: "crashing" → "crash" (base form)
        - Stopword removal: "the bug is bad" → "bug bad"
        
        This helps topic models find themes like "crash", "battery", "login"
        """
        tokens = word_tokenize(text)
        
        # Lemmatize and remove stopwords
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words 
            and token not in string.punctuation
            and len(token) > 2  # Remove very short words
        ]
        
        return tokens
    
    def preprocess(self, text):
        """
        Full preprocessing pipeline
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)
        return ' '.join(tokens)