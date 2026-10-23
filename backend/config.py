"""
Configuration settings for the application
"""

import os
from datetime import timedelta


class Config:
    """Base configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True') == 'True'
    
    # API settings
    API_TITLE = 'Customer Feedback Analyzer API'
    API_VERSION = '1.0.0'
    
    # Cache settings
    CACHE_DIR = os.environ.get('CACHE_DIR', 'cache')
    CACHE_EXPIRY = timedelta(hours=24)  # Cache expires after 24 hours
    
    # Scraping settings
    DEFAULT_REVIEW_COUNT = 500
    MAX_REVIEW_COUNT = 2000
    MIN_REVIEW_COUNT = 50
    
    # ML Model settings
    SENTIMENT_MODEL = 'nlptown/bert-base-multilingual-uncased-sentiment'
    N_TOPICS = 5
    
    # Analysis settings
    MIN_REVIEW_LENGTH = 10  # Minimum characters for a review to be analyzed
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    CACHE_EXPIRY = timedelta(hours=6)  # Shorter cache in production


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True


# Config dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}