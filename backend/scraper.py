from google_play_scraper import app, reviews, search, Sort
import pandas as pd
from datetime import datetime
import time

class PlayStoreScraper:
    """
    Scrapes Google Play Store reviews and app metadata.
    Simple and reliable implementation.
    """
    
    def search_apps(self, query):
        """
        Search for apps by name - handles missing scores
        """
        try:
            print(f"🔍 Searching for: '{query}'")
            
            # Search with more results
            results = search(query, n_hits=30)
            
            if not results:
                print(f"   No results found for '{query}'")
                return []
            
            apps = []
            seen_ids = set()
            
            for result in results:
                app_id = result.get('appId', '')
                
                # Skip if we've seen this app or if app_id is empty
                if not app_id or app_id in seen_ids:
                    continue
                
                seen_ids.add(app_id)
                
                title = result.get('title', 'Unknown App')
                
                # Simple relevance: exact match or starts with query gets priority
                query_lower = query.lower()
                title_lower = title.lower()
                
                # Calculate simple relevance score
                if title_lower == query_lower:
                    relevance = 100
                elif title_lower.startswith(query_lower):
                    relevance = 90
                elif query_lower in title_lower:
                    relevance = 80
                else:
                    # Check if any word from query is in title
                    query_words = query_lower.split()
                    matches = sum(1 for word in query_words if word in title_lower)
                    relevance = 50 + (matches * 10)
                
                # Handle missing score - default to 0
                score = result.get('score')
                if score is None:
                    score = 0
                
                apps.append({
                    'app_id': app_id,
                    'title': title,
                    'developer': result.get('developer', 'Unknown Developer'),
                    'icon': result.get('icon', 'https://via.placeholder.com/48?text=App'),
                    'score': float(score),  # Ensure it's a float
                    'relevance': relevance
                })
            
            # Sort by relevance first, then by score (both guaranteed to be numbers now)
            apps.sort(key=lambda x: (x['relevance'], x['score']), reverse=True)
            
            # Return top 15
            top_apps = apps[:15]
            
            print(f"   ✅ Found {len(top_apps)} apps")
            if top_apps:
                print(f"   Top result: {top_apps[0]['title']} (relevance: {top_apps[0]['relevance']}, score: {top_apps[0]['score']})")
            
            # Remove relevance before returning
            for app in top_apps:
                app.pop('relevance', None)
            
            return top_apps
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_app_details(self, app_id):
        """
        Fetch detailed app information
        """
        try:
            details = app(app_id, lang='en', country='us')
            return {
                'app_id': app_id,
                'title': details.get('title', 'Unknown App'),
                'developer': details.get('developer', 'Unknown Developer'),
                'category': details.get('genre', 'Unknown'),
                'rating': details.get('score', 0),
                'reviews_count': details.get('reviews', 0),
                'installs': details.get('installs', '0+'),
                'updated': details.get('updated', None),
                'icon': details.get('icon', 'https://via.placeholder.com/64?text=App')
            }
        except Exception as e:
            print(f"❌ App details error: {e}")
            return None
    
    def scrape_reviews(self, app_id, count=500):
        """
        Scrape reviews with STRICT count limiting
        """
        try:
            print(f"📊 Scraping EXACTLY {count} reviews for {app_id}...")
            
            all_reviews = []
            continuation_token = None
            
            while len(all_reviews) < count:
                remaining = count - len(all_reviews)
                batch_size = min(200, remaining)
                
                print(f"   Fetching batch... (Total so far: {len(all_reviews)}/{count})")
                
                result, continuation_token = reviews(
                    app_id,
                    lang='en',
                    country='us',
                    sort=Sort.NEWEST,
                    count=batch_size,
                    continuation_token=continuation_token
                )
                
                if not result:
                    print(f"   No more reviews available. Got {len(all_reviews)} total.")
                    break
                
                all_reviews.extend(result)
                
                if not continuation_token:
                    print(f"   Reached end of available reviews. Got {len(all_reviews)} total.")
                    break
                
                time.sleep(0.5)
            
            if not all_reviews:
                print(f"❌ No reviews found for {app_id}")
                return pd.DataFrame()
            
            df = pd.DataFrame(all_reviews)
            df = df.head(count)
            
            df['content'] = df['content'].str.strip()
            df['date'] = pd.to_datetime(df['at'])
            
            original_count = len(df)
            df = df[df['content'].str.len() > 10]
            filtered_count = original_count - len(df)
            
            if filtered_count > 0:
                print(f"   ℹ️  Filtered out {filtered_count} reviews (too short)")
            
            print(f"✅ Successfully scraped {len(df)} valid reviews (requested: {count})")
            
            return df[['reviewId', 'userName', 'content', 'score', 
                      'thumbsUpCount', 'date', 'replyContent']]
        
        except Exception as e:
            print(f"❌ Scraping error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()