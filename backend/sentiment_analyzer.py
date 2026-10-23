from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob
import re
import numpy as np

class SentimentAnalyzer:
    """
    Context-aware sentiment analysis with pattern detection
    """
    
    def __init__(self):
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.classifier = pipeline('sentiment-analysis', model=self.model, tokenizer=self.tokenizer)
            print("✅ Sentiment model loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load BERT model: {e}")
            self.classifier = None
        
        # Context patterns for sentiment correction
        self.positive_resolution_patterns = [
            r'\b(fixed|resolved|solved|working now|works? now|better now)\b',
            r'\b(issue (fixed|resolved|solved))\b',
            r'\b(bug (fixed|resolved|solved))\b',
            r'\b(problem (fixed|resolved|solved))\b',
            r'\b(finally work(s|ing)?)\b',
            r'\b(thank(s)? for (fix|update))\b',
            r'\b(great (fix|update))\b',
            r'\b(happy (now|again))\b',
            r'\b(much better)\b'
        ]
        
        self.negative_action_patterns = [
            r'\b(uninstall(ing|ed)?)\b',
            r'\b(delet(ing|ed) (the )?app)\b',
            r'\b(switch(ing|ed) to)\b',
            r'\b(moving to)\b',
            r'\b(going to uninstall)\b',
            r'\b(will uninstall)\b',
            r'\b(waste of (time|space|money))\b',
            r'\b(don\'t (download|install|use))\b',
            r'\b(avoid this app)\b'
        ]
        
        self.strong_negative_phrases = [
            r'\b(worst app ever)\b',
            r'\b(completely useless)\b',
            r'\b(total garbage)\b',
            r'\b(absolute trash)\b',
            r'\b(never work(s|ing)?)\b',
            r'\b(stopped working)\b',
            r'\b(doesn\'t work at all)\b',
            r'\b(can\'t even)\b'
        ]
        
        self.negation_patterns = [
            r'\b(not|no|never|neither|nowhere|nothing)\b',
            r'\b(doesn\'t|don\'t|didn\'t|won\'t|can\'t|couldn\'t)\b',
            r'\b(isn\'t|aren\'t|wasn\'t|weren\'t)\b'
        ]
    
    def detect_sentiment_context(self, text):
        """
        Detect context clues with proper priority handling
        
        Priority order:
        1. Conditional negatives (will/would uninstall IF...)
        2. Strong negative actions (uninstalling, deleted)
        3. Positive resolutions (fixed, resolved)
        4. Negations (not bad)
        
        Returns: (adjustment_score, reason)
        """
        text_lower = text.lower()
        
        # PRIORITY 1: Check for conditional statements (IF/UNLESS) - treat as negative
        conditional_negative_patterns = [
            r'\b(will|would|might|may)\s+(uninstall|delete|remove|switch).+(if|unless|until)\b',
            r'\b(going to|gonna)\s+(uninstall|delete).+(if|unless)\b',
            r'\b(uninstall|delete).+(if|unless|until)\b'
        ]
        
        for pattern in conditional_negative_patterns:
            if re.search(pattern, text_lower):
                return (-0.5, "Conditional threat (will uninstall if...)")
        
        # PRIORITY 2: Check for actual negative actions (stronger than resolutions)
        actual_negative_actions = [
            r'\b(uninstall(ing|ed))\b(?!\s+(fixed|resolved|issue))',  # "uninstalling" but NOT "uninstalling fixed"
            r'\b(delet(ing|ed) (the )?app)\b',
            r'\b(already (uninstalled|deleted))\b',
            r'\b(just (uninstalled|deleted))\b',
            r'\b(switch(ing|ed) to)\b(?!.*back)',  # "switching to" but NOT "switching back"
            r'\b(moving to)\b',
            r'\b(waste of (time|space|money))\b',
            r'\b(don\'t (download|install|use|bother))\b',
            r'\b(avoid this app)\b',
            r'\b((no longer|not) using)\b'
        ]
        
        for pattern in actual_negative_actions:
            if re.search(pattern, text_lower):
                return (-0.7, "Strong negative action (uninstalling/deleted/avoiding)")
        
        # PRIORITY 3: Check for positive resolutions (ONLY if no negative context)
        # Must check that issue mentions are in past tense
        positive_resolution_patterns = [
            r'\b(fixed|resolved|solved|working now|works? now)\b',
            r'\b(issue (was )?(fixed|resolved|solved))\b',
            r'\b(bug (was )?(fixed|resolved|solved))\b',
            r'\b(problem (was )?(fixed|resolved|solved))\b',
            r'\b(finally work(s|ing)?)\b',
            r'\b(thank(s)? for (fixing|fix|update))\b',
            r'\b(great (fix|update|job))\b',
            r'\b((happy|glad) (now|again))\b',
            r'\b(much better (now)?)\b',
            r'\b(improvement|improved)\b'
        ]
        
        # Only apply positive resolution if NO negative action words present
        has_negative_words = any(
            word in text_lower 
            for word in ['uninstall', 'delete', 'terrible', 'worst', 'hate', 'awful', 'useless']
        )
        
        if not has_negative_words:
            for pattern in positive_resolution_patterns:
                if re.search(pattern, text_lower):
                    return (+0.6, "Issue resolution confirmed")
    
        # PRIORITY 4: Check for strong negative phrases
        strong_negative_phrases = [
            r'\b(worst app (ever|I\'ve used))\b',
            r'\b(completely useless)\b',
            r'\b(total garbage)\b',
            r'\b(absolute(ly)? (trash|garbage|rubbish))\b',
            r'\b(never work(s|ed|ing)?)\b',
            r'\b(stopped working)\b',
            r'\b(doesn\'t work at all)\b',
            r'\b(can\'t even (open|use|start))\b',
            r'\b(piece of (trash|junk|garbage))\b'
        ]
        
        for pattern in strong_negative_phrases:
            if re.search(pattern, text_lower):
                return (-0.8, "Extremely negative phrase")
        
        # PRIORITY 5: Check for negation patterns (more nuanced)
        # "not bad" = positive, "not good" = negative
        not_bad_patterns = [
            r'\bnot (bad|terrible|awful|horrible|poor|disappointing)\b',
            r'\bnot (that|too|so) bad\b',
            r'\bnever had (issues|problems|trouble)\b'
        ]
        
        for pattern in not_bad_patterns:
            if re.search(pattern, text_lower):
                return (+0.3, "Negation of negative (not bad → positive)")
        
        not_good_patterns = [
            r'\bnot (good|great|excellent|perfect|amazing|impressive)\b',
            r'\bnot (working|work)\b',
            r'\bnot (as good|worth)\b'
        ]
        
        for pattern in not_good_patterns:
            if re.search(pattern, text_lower):
                return (-0.3, "Negation of positive (not good → negative)")
        
        # PRIORITY 6: Check for general negations with negative words
        # "can't login" = very negative
        negation_with_problem = [
            r'\b(can\'t|cannot|couldn\'t|unable to).+(login|open|use|access|connect|load)\b',
            r'\b(won\'t|doesn\'t|isn\'t).+(work|open|load|start)\b',
            r'\b(no (longer|more)).+(work|function)\b'
        ]
        
        for pattern in negation_with_problem:
            if re.search(pattern, text_lower):
                return (-0.5, "Technical problem with negation")
        
        return (0, "No context adjustment")
    
    def analyze_bert(self, text):
        """
        BERT-based sentiment with context awareness
        """
        if not self.classifier:
            return None
            
        try:
            truncated = text[:512]
            result = self.classifier(truncated)[0]
            
            label = result['label']
            confidence = result['score']
            
            if label == 'POSITIVE':
                sentiment_score = confidence * 0.8
            else:
                sentiment_score = -confidence * 0.8
            
            return {
                'label': label,
                'confidence': confidence,
                'sentiment': 'positive' if label == 'POSITIVE' else 'negative',
                'sentiment_score': sentiment_score,
                'stars': 5 if label == 'POSITIVE' else 2
            }
        except Exception as e:
            print(f"BERT sentiment error: {e}")
            return None
    
    def analyze_textblob(self, text):
        """
        TextBlob sentiment analysis
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0.3:
                sentiment = 'positive'
            elif polarity < -0.3:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment
            }
        except Exception as e:
            print(f"TextBlob sentiment error: {e}")
            return None
    
    def ensemble_sentiment(self, text):
        """
        Context-aware ensemble sentiment analysis with improved thresholds
        """
        bert_result = self.analyze_bert(text)
        blob_result = self.analyze_textblob(text)
        
        # Detect context
        context_adjustment, context_reason = self.detect_sentiment_context(text)
        
        # If BERT not available, use TextBlob only
        if not bert_result and blob_result:
            base_score = blob_result['polarity']
            adjusted_score = np.clip(base_score + context_adjustment, -1, 1)
            
            # Use tighter thresholds for more decisive classification
            if adjusted_score > 0.1:
                final_sentiment = 'positive'
            elif adjusted_score < -0.1:
                final_sentiment = 'negative'
            else:
                final_sentiment = 'neutral'
            
            return {
                'bert': None,
                'textblob': blob_result,
                'context_adjustment': context_adjustment,
                'context_reason': context_reason,
                'ensemble_score': adjusted_score,
                'final_sentiment': final_sentiment,
                'confidence': abs(adjusted_score)
            }
        
        if not bert_result and not blob_result:
            return {
                'bert': None,
                'textblob': None,
                'context_adjustment': 0,
                'context_reason': 'No analysis available',
                'ensemble_score': 0,
                'final_sentiment': 'neutral',
                'confidence': 0
            }
        
        # Both available - ensemble with context
        if bert_result and blob_result:
            # Base ensemble
            base_score = (
                0.7 * bert_result['sentiment_score'] + 
                0.3 * blob_result['polarity']
            )
            
            # Apply context adjustment
            adjusted_score = np.clip(base_score + context_adjustment, -1, 1)
            
            # Determine final sentiment with improved thresholds
            # Tighter thresholds = more decisive classification
            if adjusted_score > 0.1:
                final_sentiment = 'positive'
            elif adjusted_score < -0.1:
                final_sentiment = 'negative'
            else:
                final_sentiment = 'neutral'
            
            # Special case: if context adjustment is strong, trust it more
            if abs(context_adjustment) > 0.5:
                if adjusted_score > 0.05:
                    final_sentiment = 'positive'
                elif adjusted_score < -0.05:
                    final_sentiment = 'negative'
            
            return {
                'bert': bert_result,
                'textblob': blob_result,
                'context_adjustment': context_adjustment,
                'context_reason': context_reason,
                'ensemble_score': adjusted_score,
                'base_score': base_score,
                'final_sentiment': final_sentiment,
                'confidence': min((bert_result['confidence'] + abs(blob_result['polarity'])) / 2 + abs(context_adjustment) * 0.5, 1.0)
            }
        
        # Only BERT available
        adjusted_score = np.clip(bert_result['sentiment_score'] + context_adjustment, -1, 1)
        
        if adjusted_score > 0.1:
            final_sentiment = 'positive'
        elif adjusted_score < -0.1:
            final_sentiment = 'negative'
        else:
            final_sentiment = 'neutral'
        
        return {
            'bert': bert_result,
            'textblob': blob_result,
            'context_adjustment': context_adjustment,
            'context_reason': context_reason,
            'ensemble_score': adjusted_score,
            'final_sentiment': final_sentiment,
            'confidence': bert_result['confidence'] + abs(context_adjustment) * 0.5
        }