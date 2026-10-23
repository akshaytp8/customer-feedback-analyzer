"""
Test edge cases for sentiment analysis
"""

from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Test cases
test_reviews = [
    ("login issue resolved, thank you!", "positive"),
    ("the bug is fixed now, works great", "positive"),
    ("problem solved, much better", "positive"),
    ("finally working after the update", "positive"),
    
    ("uninstalling because of outdated data", "negative"),
    ("deleting the app, waste of time", "negative"),
    ("switching to a better app", "negative"),
    ("will uninstall if not fixed", "negative"),
    
    ("not bad, but could be better", "positive/neutral"),
    ("not good at all", "negative"),
    ("doesn't work as expected", "negative"),
    ("can't even open the app", "negative"),
]

print("🧪 Testing Sentiment Edge Cases\n")
print("=" * 80)

for review, expected in test_reviews:
    result = analyzer.ensemble_sentiment(review)
    score = result['ensemble_score']
    sentiment = result['final_sentiment']
    context = result['context_reason']
    
    match = "✅" if expected.lower() in sentiment.lower() else "❌"
    
    print(f"\n{match} Review: \"{review}\"")
    print(f"   Expected: {expected}")
    print(f"   Got: {sentiment} (score: {score:.2f})")
    print(f"   Context: {context}")
    print("-" * 80)