from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

class TopicModeler:
    """
    Extract themes/topics from reviews using multiple techniques
    
    Techniques:
    1. LDA (Latent Dirichlet Allocation): Probabilistic topic modeling
    2. NMF (Non-negative Matrix Factorization): Linear algebra approach
    3. TF-IDF + K-Means: Simpler, more interpretable clustering
    
    Why multiple techniques: Different algorithms find different patterns
    """
    
    def __init__(self, n_topics=5):
        self.n_topics = n_topics
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Limit vocabulary size
            min_df=5,  # Word must appear in at least 5 reviews
            max_df=0.8,  # Ignore words in >80% of reviews (too common)
            ngram_range=(1, 2)  # Include unigrams and bigrams (e.g., "battery drain")
        )
        
    def extract_topics_lda(self, texts):
        """
        LDA Topic Modeling
        
        How it works:
        - Assumes each review is a mix of topics
        - Each topic is a distribution over words
        - Example topics: "crash", "UI", "pricing", "battery"
        
        Why: Good at finding thematic structure
        """
        # Convert to document-term matrix
        tf_vectorizer = CountVectorizer(
            max_features=1000,
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        tf = tf_vectorizer.fit_transform(texts)
        
        # Fit LDA
        lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=50
        )
        lda.fit(tf)
        
        # Extract top words per topic
        feature_names = tf_vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]  # Top 10 words
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_id': topic_idx,
                'keywords': top_words,
                'label': self._generate_topic_label(top_words)
            })
        
        # Assign each review to dominant topic
        topic_distributions = lda.transform(tf)
        dominant_topics = topic_distributions.argmax(axis=1)
        
        return topics, dominant_topics
    
    def extract_topics_nmf(self, texts):
        """
        NMF Topic Modeling (often cleaner topics than LDA)
        
        Why NMF:
        - Often produces more interpretable topics
        - Faster than LDA
        - Works well with TF-IDF (vs. LDA with count vectors)
        """
        tfidf = self.vectorizer.fit_transform(texts)
        
        nmf = NMF(
            n_components=self.n_topics,
            random_state=42,
            max_iter=200
        )
        nmf.fit(tfidf)
        
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(nmf.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_id': topic_idx,
                'keywords': top_words,
                'label': self._generate_topic_label(top_words)
            })
        
        topic_distributions = nmf.transform(tfidf)
        dominant_topics = topic_distributions.argmax(axis=1)
        
        return topics, dominant_topics
    
    def _generate_topic_label(self, keywords):
        """
        Auto-generate human-readable topic labels
        
        Why: "Topic 0" is meaningless to PMs
        "Crash & Stability Issues" is actionable
        
        Uses keyword matching to common themes
        """
        keywords_str = ' '.join(keywords).lower()
        
        # Theme patterns (can be expanded)
        if any(word in keywords_str for word in ['crash', 'freeze', 'bug', 'error', 'hang']):
            return 'Stability & Crashes'
        elif any(word in keywords_str for word in ['battery', 'drain', 'power', 'heat']):
            return 'Battery & Performance'
        elif any(word in keywords_str for word in ['ui', 'design', 'interface', 'look', 'ugly']):
            return 'UI & Design'
        elif any(word in keywords_str for word in ['login', 'account', 'password', 'sign']):
            return 'Authentication'
        elif any(word in keywords_str for word in ['ad', 'ads', 'advertisement', 'popup']):
            return 'Ads & Monetization'
        elif any(word in keywords_str for word in ['price', 'expensive', 'cost', 'pay', 'subscription']):
            return 'Pricing'
        elif any(word in keywords_str for word in ['slow', 'lag', 'speed', 'fast', 'performance']):
            return 'Performance'
        elif any(word in keywords_str for word in ['feature', 'function', 'option', 'need']):
            return 'Feature Requests'
        else:
            # Fallback: use top 2 keywords
            return f"{keywords[0].title()} & {keywords[1].title()}"
    
    def cluster_reviews(self, texts, n_clusters=8):
        """
        K-Means clustering for alternative segmentation
        
        Why: Sometimes simpler clustering works better than topic models
        Good for finding distinct user segments
        """
        tfidf = self.vectorizer.fit_transform(texts)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=100)
        clusters = kmeans.fit_predict(tfidf)
        
        # Get cluster centers (most important words per cluster)
        feature_names = self.vectorizer.get_feature_names_out()
        cluster_themes = []
        
        for i in range(n_clusters):
            center = kmeans.cluster_centers_[i]
            top_words_idx = center.argsort()[-10:][::-1]
            top_words = [feature_names[idx] for idx in top_words_idx]
            cluster_themes.append({
                'cluster_id': i,
                'keywords': top_words,
                'label': self._generate_topic_label(top_words)
            })
        
        return cluster_themes, clusters