import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class IssueDetector:
    """
    Identifies and prioritizes customer issues for product strategy
    
    Priority factors:
    1. Frequency: How many users mention it?
    2. Recency: Is it a new issue?
    3. Sentiment: How angry are users?
    4. Upvotes: Community validation
    5. Trend: Is it getting worse?
    """
    
    def __init__(self):
        # Issue patterns (expandable with regex)
        self.issue_patterns = {
            'crash': ['crash', 'freeze', 'hang', 'stuck', 'force close', 'not responding'],
            'login': ['login', 'sign in', 'authenticate', 'password', 'locked out'],
            'battery': ['battery drain', 'battery', 'power', 'heat', 'hot'],
            'slow': ['slow', 'lag', 'loading', 'performance', 'speed'],
            'data_loss': ['lost data', 'deleted', 'disappeared', 'missing', 'gone'],
            'payment': ['charge', 'billing', 'payment', 'refund', 'subscription'],
            'ui_bug': ['button not work', 'cant click', 'ui bug', 'broken'],
            'ads': ['too many ads', 'ad spam', 'popup', 'advertisement']
        }
    
    def detect_issues(self, df):
        """
        Scan reviews for known issue patterns
        
        Returns DataFrame with issue columns (binary)
        """
        df_copy = df.copy()
        
        for issue_type, patterns in self.issue_patterns.items():
            # Check if review mentions this issue
            pattern_regex = '|'.join(patterns)
            df_copy[f'issue_{issue_type}'] = df_copy['content'].str.contains(
                pattern_regex, case=False, na=False
            ).astype(int)
        
        return df_copy
    
    def rank_issues(self, df):
        """
        Priority ranking algorithm for issues
        
        Score = (Frequency × 0.3) + (Recency × 0.25) + 
                (Severity × 0.25) + (Community_Impact × 0.2)
        
        Why these weights:
        - Frequency matters most (widespread problem)
        - Recency important (new bugs from recent update)
        - Severity (angry users = bigger problem)
        - Community impact (high upvotes = validated concern)
        """
        issues = []
        
        for issue_type in self.issue_patterns.keys():
            col = f'issue_{issue_type}'
            if col not in df.columns:
                continue
            
            issue_df = df[df[col] == 1]
            
            if len(issue_df) == 0:
                continue
            
            # Calculate metrics
            frequency = len(issue_df)
            frequency_pct = frequency / len(df)
            
            # Recency: Average days since mentioned
            if 'date' in df.columns:
                days_ago = (datetime.now() - issue_df['date']).dt.days.mean()
                recency_score = max(0, 1 - (days_ago / 30))  # Decay over 30 days
            else:
                recency_score = 0.5
            
            # Severity: Average sentiment of issue mentions
            if 'sentiment_score' in issue_df.columns:
                severity_score = -issue_df['sentiment_score'].mean()  # More negative = higher severity
                severity_score = max(0, min(1, (severity_score + 1) / 2))  # Normalize to 0-1
            else:
                severity_score = 0.5
            
            # Community impact: Average thumbs up
            if 'thumbsUpCount' in issue_df.columns:
                avg_upvotes = issue_df['thumbsUpCount'].mean()
                community_score = min(1, avg_upvotes / 50)  # Cap at 50 upvotes
            else:
                community_score = 0
            
            # Calculate priority score
            priority_score = (
                frequency_pct * 0.3 +
                recency_score * 0.25 +
                severity_score * 0.25 +
                community_score * 0.2
            ) * 100
            
            issues.append({
                'issue_type': issue_type.replace('_', ' ').title(),
                'mentions': frequency,
                'percentage': frequency_pct * 100,
                'recency_score': recency_score,
                'severity_score': severity_score,
                'community_score': community_score,
                'priority_score': priority_score,
                'sample_reviews': issue_df['content'].head(3).tolist()
            })
        
        # Sort by priority
        issues_df = pd.DataFrame(issues).sort_values('priority_score', ascending=False)
        return issues_df
    
    def trend_analysis(self, df, window_days=7):
        """
        Detect trending issues (getting worse over time)
        
        Why: A spike in crashes after v3.2 update is critical info
        """
        if 'date' not in df.columns:
            return pd.DataFrame()
        
        df = df.sort_values('date')
        trends = []
        
        for issue_type in self.issue_patterns.keys():
            col = f'issue_{issue_type}'
            if col not in df.columns:
                continue
            
            # Count issues per week
            df_issue = df[df[col] == 1].copy()
            df_issue['week'] = df_issue['date'].dt.to_period('W')
            weekly_counts = df_issue.groupby('week').size()
            
            if len(weekly_counts) < 2:
                continue
            
            # Calculate trend (linear regression slope)
            x = np.arange(len(weekly_counts))
            y = weekly_counts.values
            slope = np.polyfit(x, y, 1)[0]
            
            # Positive slope = increasing problem
            if slope > 0:
                trends.append({
                    'issue': issue_type.replace('_', ' ').title(),
                    'trend': 'Increasing' if slope > 0 else 'Decreasing',
                    'change_rate': slope,
                    'recent_count': weekly_counts.iloc[-1] if len(weekly_counts) > 0 else 0
                })
        
        return pd.DataFrame(trends).sort_values('change_rate', ascending=False)