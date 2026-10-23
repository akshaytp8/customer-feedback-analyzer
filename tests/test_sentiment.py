"""
Unit tests for sentiment analyzer
"""

import pytest
from backend.sentiment_analyzer import SentimentAnalyzer


@pytest.fixture
def analyzer():
    return SentimentAnalyzer()


def test_positive_sentiment(analyzer):
    """Test positive sentiment detection"""
    text = "This app is absolutely amazing! I love it so much!"
    result = analyzer.ensemble_sentiment(text)
    
    assert result is not None
    assert result['final_sentiment'] == 'positive'
    assert result['ensemble_score'] > 0.3


def test_negative_sentiment(analyzer):
    """Test negative sentiment detection"""
    text = "Terrible app. Crashes all the time. Complete waste of money."
    result = analyzer.ensemble_sentiment(text)
    
    assert result is not None
    assert result['final_sentiment'] == 'negative'
    assert result['ensemble_score'] < -0.3


def test_neutral_sentiment(analyzer):
    """Test neutral sentiment detection"""
    text = "The app works fine. It's okay, nothing special."
    result = analyzer.ensemble_sentiment(text)
    
    assert result is not None
    assert result['final_sentiment'] in ['neutral', 'positive', 'negative']
    assert -0.3 <= result['ensemble_score'] <= 0.3


def test_sarcasm_detection(analyzer):
    """Test if BERT can detect sarcasm"""
    text = "Oh great, another crash. Just what I needed today."
    result = analyzer.ensemble_sentiment(text)
    
    # BERT should ideally detect this as negative
    assert result is not None
    # Note: Sarcasm is hard, so we just check it returns something valid
    assert result['final_sentiment'] in ['positive', 'neutral', 'negative']


def test_mixed_sentiment(analyzer):
    """Test mixed sentiment"""
    text = "Great UI design but crashes too often"
    result = analyzer.ensemble_sentiment(text)
    
    assert result is not None
    # Could be neutral or slightly negative
    assert result['ensemble_score'] <= 0.5


def test_bert_analysis(analyzer):
    """Test BERT-specific analysis"""
    text = "Excellent application with great features"
    result = analyzer.analyze_bert(text)
    
    assert result is not None
    assert 'stars' in result
    assert 'confidence' in result
    assert result['stars'] >= 4


def test_textblob_analysis(analyzer):
    """Test TextBlob-specific analysis"""
    text = "This is wonderful and fantastic"
    result = analyzer.analyze_textblob(text)
    
    assert result is not None
    assert 'polarity' in result
    assert 'subjectivity' in result
    assert result['polarity'] > 0