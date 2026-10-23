"""
Main Flask Application
Handles API endpoints for the Customer Feedback Analyzer
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import logging
from datetime import datetime
import os
import pickle
import traceback
import math

from scraper import PlayStoreScraper
from preprocessing import TextPreprocessor
from sentiment_analyzer import SentimentAnalyzer
from topic_modeler import TopicModeler
from issue_detector import IssueDetector
from config import Config
from utils import create_cache_key, format_error_response

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config.from_object(Config)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize services
scraper = PlayStoreScraper()
preprocessor = TextPreprocessor()
sentiment_analyzer = SentimentAnalyzer()
topic_modeler = TopicModeler(n_topics=5)
issue_detector = IssueDetector()

# Cache directory
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    Returns: API status and version
    """
    return jsonify(sanitize_for_json({
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    }))


@app.route('/api/search', methods=['POST'])
def search_apps():
    """
    Search for apps by name on Google Play Store
    
    Request Body:
        {
            "query": "instagram"
        }
    
    Response:
        {
            "success": true,
            "apps": [
                {
                    "app_id": "com.instagram.android",
                    "title": "Instagram",
                    "developer": "Instagram",
                    "icon": "https://...",
                    "score": 4.3
                }
            ],
            "count": 10
        }
    """
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify(format_error_response('Query parameter is required')), 400
        
        if len(query) < 2:
            return jsonify(format_error_response('Query must be at least 2 characters')), 400
        
        logger.info(f"Searching for apps with query: {query}")
        apps = scraper.search_apps(query)
        
        return jsonify(sanitize_for_json({
            'success': True,
            'apps': apps,
            'count': len(apps),
            'timestamp': datetime.now().isoformat()
        }))
    
    except Exception as e:
        logger.error(f"Search error: {str(e)}\n{traceback.format_exc()}")
        return jsonify(format_error_response(str(e))), 500


@app.route('/api/app-details', methods=['POST'])
def get_app_details():
    """
    Get detailed information about a specific app
    
    Request Body:
        {
            "app_id": "com.instagram.android"
        }
    
    Response:
        {
            "success": true,
            "app_details": {
                "app_id": "com.instagram.android",
                "title": "Instagram",
                "developer": "Instagram",
                "category": "Social",
                "rating": 4.3,
                "reviews_count": 50000000,
                "installs": "1,000,000,000+",
                "updated": 1699564800
            }
        }
    """
    try:
        data = request.json
        app_id = data.get('app_id', '').strip()
        
        if not app_id:
            return jsonify(format_error_response('app_id is required')), 400
        
        logger.info(f"Fetching details for app: {app_id}")
        app_details = scraper.get_app_details(app_id)
        
        if not app_details:
            return jsonify(format_error_response('App not found')), 404
        
        return jsonify(sanitize_for_json({
            'success': True,
            'app_details': app_details,
            'timestamp': datetime.now().isoformat()
        }))
    
    except Exception as e:
        logger.error(f"App details error: {str(e)}\n{traceback.format_exc()}")
        return jsonify(format_error_response(str(e))), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_app():
    """
    Complete analysis pipeline for an app
    
    Request Body:
        {
            "app_id": "com.instagram.android",
            "review_count": 500,
            "force_refresh": false
        }
    
    Response:
        {
            "success": true,
            "app_details": {...},
            "total_reviews": 500,
            "sentiment_distribution": {...},
            "topics_lda": [...],
            "topics_nmf": [...],
            "top_issues": [...],
            "trending_issues": [...],
            "insights": {...},
            "recent_reviews": [...],
            "time_series": [...]
        }
    """
    try:
        data = request.json
        app_id = data.get('app_id', '').strip()
        review_count = data.get('review_count', 500)
        force_refresh = data.get('force_refresh', False)
        
        if not app_id:
            return jsonify(format_error_response('app_id is required')), 400
        
        if review_count < 50 or review_count > 2000:
            return jsonify(format_error_response('review_count must be between 50 and 2000')), 400
        
        # Check cache
        cache_key = create_cache_key(app_id, review_count)
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
        
        if not force_refresh and os.path.exists(cache_file):
            logger.info(f"Loading cached analysis for {app_id}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                cached_data['from_cache'] = True
                return jsonify(sanitize_for_json(cached_data))
        
        logger.info(f"Starting fresh analysis for {app_id} with {review_count} reviews")
        
        # Step 1: Get app details
        app_details = scraper.get_app_details(app_id)
        if not app_details:
            return jsonify(format_error_response('App not found')), 404
        
        # Step 2: Scrape reviews
        logger.info("Scraping reviews...")
        reviews_df = scraper.scrape_reviews(app_id, count=review_count)
        
        if reviews_df.empty:
            return jsonify(format_error_response('No reviews found for this app')), 404
        
        logger.info(f"Scraped {len(reviews_df)} reviews")
        
        # Step 3: Preprocess text
        logger.info("Preprocessing reviews...")
        reviews_df['cleaned_content'] = reviews_df['content'].apply(
            preprocessor.preprocess
        )
        
        # Step 4: Sentiment analysis
        logger.info("Analyzing sentiment...")
        sentiment_results = []
        for idx, text in enumerate(reviews_df['content']):
            if idx % 50 == 0:
                logger.info(f"Processed {idx}/{len(reviews_df)} reviews for sentiment")
            result = sentiment_analyzer.ensemble_sentiment(text)
            sentiment_results.append(result)
        
        reviews_df['sentiment_score'] = [r['ensemble_score'] for r in sentiment_results]
        reviews_df['sentiment'] = [r['final_sentiment'] for r in sentiment_results]
        reviews_df['sentiment_confidence'] = [r['confidence'] for r in sentiment_results]
        
        # Step 5: Topic modeling
        logger.info("Extracting topics with LDA...")
        topics_lda, dominant_topics_lda = topic_modeler.extract_topics_lda(
            reviews_df['cleaned_content'].tolist()
        )
        
        logger.info("Extracting topics with NMF...")
        topics_nmf, dominant_topics_nmf = topic_modeler.extract_topics_nmf(
            reviews_df['cleaned_content'].tolist()
        )
        
        reviews_df['topic_lda'] = dominant_topics_lda
        reviews_df['topic_nmf'] = dominant_topics_nmf
        
        # Step 6: Issue detection
        logger.info("Detecting issues...")
        reviews_df = issue_detector.detect_issues(reviews_df)
        issues = issue_detector.rank_issues(reviews_df)
        trends = issue_detector.trend_analysis(reviews_df)
        
        # Step 7: Generate insights
        logger.info("Generating insights...")
        insights = generate_insights(reviews_df, app_details, issues, topics_nmf)
        
        # Prepare response
        response = {
            'success': True,
            'from_cache': False,
            'app_details': app_details,
            'total_reviews': len(reviews_df),
            'sentiment_distribution': {
                'positive': int((reviews_df['sentiment'] == 'positive').sum()),
                'neutral': int((reviews_df['sentiment'] == 'neutral').sum()),
                'negative': int((reviews_df['sentiment'] == 'negative').sum()),
                'avg_score': float(reviews_df['sentiment_score'].mean()),
                'score_std': float(reviews_df['sentiment_score'].std()),
                'median_score': float(reviews_df['sentiment_score'].median())
            },
            'topics_lda': topics_lda,
            'topics_nmf': topics_nmf,
            'top_issues': issues.head(10).to_dict('records') if not issues.empty else [],
            'trending_issues': trends.to_dict('records') if not trends.empty else [],
            'insights': insights,
            'recent_reviews': prepare_recent_reviews(reviews_df),
            'time_series': prepare_time_series(reviews_df),
            'rating_distribution': prepare_rating_distribution(reviews_df),
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache results
        logger.info("Caching results...")
        with open(cache_file, 'wb') as f:
            pickle.dump(response, f)
        
        logger.info("Analysis complete!")
        return jsonify(sanitize_for_json(response))
    
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}\n{traceback.format_exc()}")
        return jsonify(format_error_response(str(e))), 500


@app.route('/api/export', methods=['POST'])
def export_data():
    """
    Export analysis data for sharing
    
    Request Body:
        {
            "app_id": "com.instagram.android"
        }
    
    Response:
        {
            "success": true,
            "export_data": {
                "app": "Instagram",
                "health_score": 75.3,
                "total_reviews": 500,
                ...
            }
        }
    """
    try:
        data = request.json
        app_id = data.get('app_id', '')
        
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.startswith(app_id)]
        
        if not cache_files:
            return jsonify(format_error_response('No analysis found. Please run analysis first.')), 404
        
        # Load most recent cache
        cache_file = os.path.join(CACHE_DIR, cache_files[-1])
        with open(cache_file, 'rb') as f:
            analysis = pickle.load(f)
        
        # Create export summary
        export_data = {
            'app': analysis['app_details']['title'],
            'developer': analysis['app_details']['developer'],
            'category': analysis['app_details']['category'],
            'health_score': analysis['insights']['health_score'],
            'health_status': analysis['insights']['health_status'],
            'total_reviews_analyzed': analysis['total_reviews'],
            'sentiment_positive': analysis['sentiment_distribution']['positive'],
            'sentiment_neutral': analysis['sentiment_distribution']['neutral'],
            'sentiment_negative': analysis['sentiment_distribution']['negative'],
            'avg_sentiment_score': analysis['sentiment_distribution']['avg_score'],
            'top_issue': analysis['insights']['top_priority']['issue'] if analysis['top_issues'] else 'None',
            'top_issue_severity': analysis['insights']['top_priority']['severity'] if analysis['top_issues'] else 'N/A',
            'affected_users': analysis['insights']['top_priority']['affected_users'] if analysis['top_issues'] else '0%',
            'recommendation': analysis['insights']['top_priority']['recommendation'] if analysis['top_issues'] else 'No critical issues',
            'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(sanitize_for_json({
            'success': True,
            'export_data': export_data
        }))
    
    except Exception as e:
        logger.error(f"Export error: {str(e)}\n{traceback.format_exc()}")
        return jsonify(format_error_response(str(e))), 500


@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """
    Clear cached analysis data
    
    Request Body:
        {
            "app_id": "com.instagram.android"  // Optional: clear specific app
        }
    """
    try:
        data = request.json
        app_id = data.get('app_id', None)
        
        if app_id:
            # Clear specific app cache
            cache_files = [f for f in os.listdir(CACHE_DIR) if f.startswith(app_id)]
            for cache_file in cache_files:
                os.remove(os.path.join(CACHE_DIR, cache_file))
            cleared_count = len(cache_files)
        else:
            # Clear all cache
            cache_files = os.listdir(CACHE_DIR)
            for cache_file in cache_files:
                os.remove(os.path.join(CACHE_DIR, cache_file))
            cleared_count = len(cache_files)
        
        return jsonify(sanitize_for_json({
            'success': True,
            'message': f'Cleared {cleared_count} cached analyses',
            'cleared_count': cleared_count
        }))
    
    except Exception as e:
        logger.error(f"Clear cache error: {str(e)}")
        return jsonify(format_error_response(str(e))), 500


def generate_insights(df, app_details, issues, topics):
    """
    Generate actionable insights for PMs
    """
    insights = {}
    
    # Overall health score (0-100)
    sentiment_health = (df['sentiment_score'].mean() + 1) * 50
    rating_health = (df['score'].mean() / 5) * 100
    health_score = (sentiment_health * 0.6) + (rating_health * 0.4)
    
    insights['health_score'] = round(health_score, 1)
    insights['health_status'] = (
        'Excellent' if health_score > 80 else
        'Good' if health_score > 60 else
        'Fair' if health_score > 40 else
        'Poor'
    )
    
    # Top priority action
    if not issues.empty:
        top_issue = issues.iloc[0]
        insights['top_priority'] = {
            'issue': top_issue['issue_type'],
            'severity': 'Critical' if top_issue['priority_score'] > 70 else 'High' if top_issue['priority_score'] > 50 else 'Medium',
            'affected_users': f"{top_issue['percentage']:.1f}%",
            'recommendation': generate_recommendation(top_issue),
            'sample_reviews': top_issue['sample_reviews']
        }
    else:
        insights['top_priority'] = {
            'issue': 'None',
            'severity': 'N/A',
            'affected_users': '0%',
            'recommendation': 'No critical issues detected. Continue monitoring user feedback.',
            'sample_reviews': []
        }
    
    # Sentiment trend
    if 'date' in df.columns:
        recent_df = df[df['date'] > (df['date'].max() - pd.Timedelta(days=7))]
        older_df = df[df['date'] <= (df['date'].max() - pd.Timedelta(days=7))]
        
        if len(recent_df) > 0 and len(older_df) > 0:
            recent_sentiment = recent_df['sentiment_score'].mean()
            older_sentiment = older_df['sentiment_score'].mean()
            change = recent_sentiment - older_sentiment
            
            insights['sentiment_trend'] = {
                'direction': 'improving' if change > 0.05 else 'declining' if change < -0.05 else 'stable',
                'change_percent': round(change * 100, 1),
                'recent_avg': round(recent_sentiment, 2),
                'previous_avg': round(older_sentiment, 2)
            }
        else:
            insights['sentiment_trend'] = {
                'direction': 'insufficient_data',
                'change_percent': 0,
                'recent_avg': 0,
                'previous_avg': 0
            }
    
    # Topic distribution
    topic_distribution = {}
    for topic in topics:
        topic_label = topic['label']
        count = (df['topic_nmf'] == topic['topic_id']).sum()
        
        # Get average sentiment for this topic
        topic_reviews = df[df['topic_nmf'] == topic['topic_id']]
        avg_sentiment = topic_reviews['sentiment_score'].mean() if len(topic_reviews) > 0 else 0
        
        topic_distribution[topic_label] = {
            'count': int(count),
            'percentage': round((count / len(df)) * 100, 1),
            'avg_sentiment': round(avg_sentiment, 2),
            'sentiment_label': 'positive' if avg_sentiment > 0.2 else 'negative' if avg_sentiment < -0.2 else 'neutral'
        }
    
    insights['topic_focus_areas'] = topic_distribution
    
    # Key strengths and weaknesses
    insights['strengths'] = identify_strengths(df)
    insights['weaknesses'] = identify_weaknesses(df, issues)
    
    return insights


def generate_recommendation(issue_row):
    """
    Generate actionable recommendations
    """
    issue_type = issue_row['issue_type'].lower()
    severity = issue_row['priority_score']
    mentions = issue_row['mentions']
    percentage = issue_row['percentage']
    
    recommendations = {
        'crash': f"🚨 URGENT: Investigate crash logs immediately. {mentions} users affected ({percentage:.1f}%). Review recent code changes, check error monitoring tools, and consider emergency hotfix if crash rate is high.",
        'stability': f"⚠️ HIGH PRIORITY: App stability issues detected. Run comprehensive QA testing, review recent deployments, and check for memory leaks or race conditions.",
        'battery drain': f"⚡ Audit background processes and network calls. Profile app with Android/iOS profilers to identify power-intensive operations. {percentage:.1f}% of users impacted.",
        'battery': f"⚡ Battery performance concern. Review location services usage, background sync frequency, and wake locks. Consider battery optimization best practices.",
        'login': f"🔐 Authentication issues reported. Check auth service uptime, review rate limiting, validate token refresh logic, and test on various network conditions.",
        'sign in': f"🔐 Sign-in problems detected. Verify SSO integrations, check session management, and ensure password reset flows work correctly.",
        'slow': f"🐌 Performance audit needed ({mentions} complaints). Profile app performance, optimize API response times, implement caching, and review database query efficiency.",
        'performance': f"⚡ Speed issues detected. Run performance profiling, optimize image loading, minimize main thread work, and consider lazy loading for heavy components.",
        'lag': f"🎮 UI lag reported. Check for main thread blocking, optimize animations, reduce overdraw, and profile frame rendering times.",
        'ui bug': f"🐛 UI defect detected. QA regression testing required. Document bug with screenshots, test on various screen sizes and OS versions.",
        'button': f"🖱️ Interactive element malfunction. Test touch targets, verify click handlers, and check for z-index or overlay issues.",
        'ads': f"📊 Ad experience concerns. Review ad frequency (current impact: {mentions} users), test ad placement, ensure ads don't block critical UI, and balance monetization vs UX.",
        'advertisement': f"📺 Advertisement feedback. Consider frequency capping, improve ad relevance, and provide clear options for ad-free experience.",
        'payment': f"💳 CRITICAL: Payment issues! Escalate to payments team immediately. Audit transaction logs, check payment gateway status. Potential revenue impact: {mentions} affected users.",
        'billing': f"💰 Billing concerns raised. Review subscription logic, verify receipt validation, and ensure clear pricing communication.",
        'data loss': f"🔴 CRITICAL: Data loss reported! Immediate investigation required. Check data persistence logic, backup systems, and sync mechanisms. This severely impacts user trust.",
        'lost data': f"🔴 URGENT: Users reporting lost data. Audit data storage, implement data recovery options, and review sync conflict resolution."
    }
    
    for key, rec in recommendations.items():
        if key in issue_type:
            return rec
    
    # Default recommendation
    return f"⚙️ Investigate {mentions} user reports about '{issue_row['issue_type']}'. Priority score: {severity:.1f}/100. Review sample feedback and create action plan."


def identify_strengths(df):
    """
    Identify what users love about the app
    """
    positive_reviews = df[df['sentiment_score'] > 0.5]
    
    if len(positive_reviews) == 0:
        return []
    
    # Common positive words
    from collections import Counter
    positive_text = ' '.join(positive_reviews['cleaned_content'])
    words = positive_text.split()
    
    # Filter for meaningful words (simple approach)
    positive_keywords = ['great', 'love', 'awesome', 'excellent', 'perfect', 
                        'amazing', 'best', 'fantastic', 'wonderful', 'easy',
                        'simple', 'fast', 'smooth', 'beautiful', 'clean']
    
    found_strengths = [word for word in words if word in positive_keywords]
    strength_counts = Counter(found_strengths)
    
    strengths = []
    for word, count in strength_counts.most_common(3):
        percentage = (count / len(positive_reviews)) * 100
        strengths.append({
            'attribute': word.title(),
            'mentions': count,
            'percentage': round(percentage, 1)
        })
    
    return strengths


def identify_weaknesses(df, issues_df):
    """
    Identify main pain points
    """
    if issues_df.empty:
        return []
    
    weaknesses = []
    for idx, issue in issues_df.head(3).iterrows():
        weaknesses.append({
            'issue': issue['issue_type'],
            'severity': issue['priority_score'],
            'affected_users': f"{issue['percentage']:.1f}%"
        })
    
    return weaknesses


def prepare_recent_reviews(df):
    """
    Format recent reviews for frontend
    """
    recent = df.nlargest(20, 'date')[
        ['content', 'score', 'sentiment', 'sentiment_score', 'date', 'thumbsUpCount']
    ].copy()
    
    recent['date'] = recent['date'].dt.strftime('%Y-%m-%d')
    recent['sentiment_score'] = recent['sentiment_score'].round(2)
    
    return recent.to_dict('records')


def prepare_time_series(df):
    """
    Aggregate data for time-series charts with better readability
    
    Strategy:
    - Group by day (or week if >90 days of data)
    - Calculate daily averages
    - Smooth the trend line
    """
    if 'date' not in df.columns or len(df) == 0:
        return []
    
    df = df.copy()
    df['date_only'] = df['date'].dt.date
    
    # Determine date range
    date_range = (df['date'].max() - df['date'].min()).days
    
    # Choose aggregation period based on data range
    if date_range > 180:
        # For >6 months data, group by week
        df['period'] = df['date'].dt.to_period('W')
        period_label = 'Week'
    elif date_range > 60:
        # For >2 months data, group by week
        df['period'] = df['date'].dt.to_period('W')
        period_label = 'Week'
    else:
        # For <2 months data, group by day
        df['period'] = df['date'].dt.to_period('D')
        period_label = 'Day'
    
    # Aggregate by period
    time_series = df.groupby('period').agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'score': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    time_series.columns = ['period', 'sentiment_mean', 'sentiment_std', 'volume', 
                           'rating_mean', 'rating_std']
    
    # Convert period to string date
    time_series['date'] = time_series['period'].dt.start_time.dt.strftime('%Y-%m-%d')
    
    # Apply moving average for smoother trend (3-period window)
    time_series['sentiment_smooth'] = time_series['sentiment_mean'].rolling(
        window=3, center=True, min_periods=1
    ).mean()
    
    time_series['rating_smooth'] = time_series['rating_mean'].rolling(
        window=3, center=True, min_periods=1
    ).mean()
    
    # Round for cleaner display
    time_series['sentiment'] = time_series['sentiment_smooth'].round(3)
    time_series['rating'] = time_series['rating_smooth'].round(2)
    time_series['volume'] = time_series['volume'].astype(int)
    
    # Add confidence intervals (for advanced visualization)
    time_series['sentiment_ci_lower'] = (
        time_series['sentiment_mean'] - time_series['sentiment_std']
    ).round(3)
    time_series['sentiment_ci_upper'] = (
        time_series['sentiment_mean'] + time_series['sentiment_std']
    ).round(3)
    
    return time_series[['date', 'sentiment', 'rating', 'volume', 
                        'sentiment_ci_lower', 'sentiment_ci_upper']].to_dict('records')


def prepare_rating_distribution(df):
    """
    Get distribution of star ratings (1-5)
    """
    rating_counts = df['score'].value_counts().sort_index()
    
    distribution = []
    for rating in range(1, 6):
        count = rating_counts.get(rating, 0)
        distribution.append({
            'rating': rating,
            'count': int(count),
            'percentage': round((count / len(df)) * 100, 1)
        })
    
    return distribution


def sanitize_for_json(obj):
    """
    Recursively sanitize objects for JSON serialization
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    else:
        return obj


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify(format_error_response('Endpoint not found')), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify(format_error_response('Internal server error')), 500


if __name__ == '__main__':
    logger.info("Starting Customer Feedback Analyzer API...")
    logger.info(f"Cache directory: {os.path.abspath(CACHE_DIR)}")
    app.run(debug=True, host='0.0.0.0', port=5001)