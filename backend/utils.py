"""
Utility functions for the application
"""

import hashlib
import json
from datetime import datetime


def create_cache_key(app_id, review_count):
    """
    Create a unique cache key for an analysis
    
    Args:
        app_id: Google Play app ID
        review_count: Number of reviews analyzed
    
    Returns:
        Hash string for cache filename
    """
    key_string = f"{app_id}_{review_count}"
    return hashlib.md5(key_string.encode()).hexdigest()


def format_error_response(message, error_code=None):
    """
    Format error responses consistently
    
    Args:
        message: Error message
        error_code: Optional error code
    
    Returns:
        Dictionary with error information
    """
    response = {
        'success': False,
        'error': message,
        'timestamp': datetime.now().isoformat()
    }
    
    if error_code:
        response['error_code'] = error_code
    
    return response


def validate_app_id(app_id):
    """
    Validate Google Play app ID format
    
    Args:
        app_id: App ID to validate
    
    Returns:
        Boolean indicating if valid
    """
    if not app_id or not isinstance(app_id, str):
        return False
    
    # Basic validation: should contain at least one dot and no spaces
    if '.' not in app_id or ' ' in app_id:
        return False
    
    return True


def calculate_percentage(count, total):
    """
    Calculate percentage safely
    
    Args:
        count: Numerator
        total: Denominator
    
    Returns:
        Percentage as float, or 0 if total is 0
    """
    if total == 0:
        return 0.0
    return round((count / total) * 100, 2)


def truncate_text(text, max_length=100):
    """
    Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
    
    Returns:
        Truncated text with ellipsis if needed
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length] + '...'


def parse_date(date_str):
    """
    Parse date string to datetime object
    
    Args:
        date_str: Date string in various formats
    
    Returns:
        datetime object or None
    """
    from dateutil import parser
    try:
        return parser.parse(date_str)
    except:
        return None


def safe_divide(numerator, denominator, default=0):
    """
    Safe division that handles zero denominator
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails
    
    Returns:
        Result of division or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default


def format_number(number, decimals=2):
    """
    Format number with comma separators and decimals
    
    Args:
        number: Number to format
        decimals: Number of decimal places
    
    Returns:
        Formatted string
    """
    if number is None:
        return 'N/A'
    
    if isinstance(number, int):
        return f"{number:,}"
    
    return f"{number:,.{decimals}f}"


def merge_dicts(*dicts):
    """
    Merge multiple dictionaries
    
    Args:
        *dicts: Variable number of dictionaries
    
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result