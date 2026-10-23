"""
Unit tests for scraper module
"""

import pytest
from backend.scraper import PlayStoreScraper


@pytest.fixture
def scraper():
    return PlayStoreScraper()


def test_search_apps(scraper):
    """Test app search functionality"""
    results = scraper.search_apps("instagram")
    
    assert isinstance(results, list)
    assert len(results) > 0
    
    # Check first result structure
    app = results[0]
    assert 'app_id' in app
    assert 'title' in app
    assert 'developer' in app
    assert 'icon' in app
    assert 'score' in app


def test_search_apps_no_results(scraper):
    """Test search with query that returns no results"""
    results = scraper.search_apps("xyzabcnonexistentapp12345")
    assert isinstance(results, list)
    # May return empty list or no results


def test_get_app_details(scraper):
    """Test fetching app details"""
    # Use a known app ID
    app_id = "com.instagram.android"
    details = scraper.get_app_details(app_id)
    
    assert details is not None
    assert details['app_id'] == app_id
    assert 'title' in details
    assert 'developer' in details
    assert 'rating' in details
    assert details['rating'] >= 0 and details['rating'] <= 5


def test_get_app_details_invalid(scraper):
    """Test fetching details for invalid app"""
    details = scraper.get_app_details("invalid.app.id")
    assert details is None


def test_scrape_reviews(scraper):
    """Test review scraping"""
    app_id = "com.instagram.android"
    df = scraper.scrape_reviews(app_id, count=10)
    
    assert not df.empty
    assert len(df) > 0
    assert 'content' in df.columns
    assert 'score' in df.columns
    assert 'date' in df.columns


def test_scrape_reviews_filtering(scraper):
    """Test that short reviews are filtered out"""
    app_id = "com.instagram.android"
    df = scraper.scrape_reviews(app_id, count=50)
    
    # All reviews should be longer than 10 characters
    assert all(df['content'].str.len() > 10)