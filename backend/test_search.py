# Create test file: backend/test_search.py
from google_play_scraper import search

# Test different queries
queries = ['ticker', 'tickertape', 'tickertape stocks']

for query in queries:
    print(f"\n🔍 Testing: '{query}'")
    results = search(query, n_hits=10)
    
    if results:
        print(f"   ✅ Found {len(results)} results")
        for i, result in enumerate(results[:3]):
            print(f"   {i+1}. {result['title']}")
    else:
        print(f"   ❌ No results")