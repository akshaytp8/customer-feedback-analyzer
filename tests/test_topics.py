"""
Unit tests for topic modeler
"""

import pytest
from backend.topic_modeler import TopicModeler


@pytest.fixture
def topic_modeler():
    return TopicModeler(n_topics=3)


@pytest.fixture
def sample_reviews():
    return [
        "The app crashes all the time, very frustrating",
        "Battery drain is a huge problem, phone gets hot",
        "Love the new UI design, looks beautiful",
        "Interface is clean and easy to use",
        "Constant crashes make it unusable",
        "Great features but drains battery too fast",
        "Beautiful design and smooth animations",
        "App keeps freezing and crashing",
        "UI is gorgeous but performance is terrible",
        "Battery life is awful, gets very hot"
    ]


def test_lda_topic_extraction(topic_modeler, sample_reviews):
    """Test LDA topic modeling"""
    topics, dominant_topics = topic_modeler.extract_topics_lda(sample_reviews)
    
    assert len(topics) == 3
    assert len(dominant_topics) == len(sample_reviews)
    
    # Check topic structure
    for topic in topics:
        assert 'topic_id' in topic
        assert 'keywords' in topic
        assert 'label' in topic
        assert len(topic['keywords']) > 0


def test_nmf_topic_extraction(topic_modeler, sample_reviews):
    """Test NMF topic modeling"""
    topics, dominant_topics = topic_modeler.extract_topics_nmf(sample_reviews)
    
    assert len(topics) == 3
    assert len(dominant_topics) == len(sample_reviews)
    
    # Check topic structure
    for topic in topics:
        assert 'topic_id' in topic
        assert 'keywords' in topic
        assert 'label' in topic


def test_topic_label_generation(topic_modeler):
    """Test automatic topic label generation"""
    keywords_crash = ['crash', 'freeze', 'bug', 'error']
    label = topic_modeler._generate_topic_label(keywords_crash)
    assert 'Stability' in label or 'Crash' in label
    
    keywords_battery = ['battery', 'drain', 'power', 'heat']
    label = topic_modeler._generate_topic_label(keywords_battery)
    assert 'Battery' in label or 'Performance' in label
    
    keywords_ui = ['ui', 'design', 'interface', 'look']
    label = topic_modeler._generate_topic_label(keywords_ui)
    assert 'UI' in label or 'Design' in label


def test_clustering(topic_modeler, sample_reviews):
    """Test K-means clustering"""
    cluster_themes, clusters = topic_modeler.cluster_reviews(sample_reviews, n_clusters=3)
    
    assert len(cluster_themes) == 3
    assert len(clusters) == len(sample_reviews)
    
    # Check cluster structure
    for theme in cluster_themes:
        assert 'cluster_id' in theme
        assert 'keywords' in theme
        assert 'label' in theme


def test_empty_input(topic_modeler):
    """Test handling of empty input"""
    with pytest.raises(Exception):
        topic_modeler.extract_topics_lda([])