# Backend API

Flask-based REST API for customer feedback analysis.

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## API Endpoints

### Health Check
```
GET /api/health
```

### Search Apps
```
POST /api/search
Body: { "query": "instagram" }
```

### Analyze App
```
POST /api/analyze
Body: {
  "app_id": "com.instagram.android",
  "review_count": 500,
  "force_refresh": false
}
```

### Export Data
```
POST /api/export
Body: { "app_id": "com.instagram.android" }
```

## Architecture

- `app.py`: Flask routes
- `scraper.py`: Google Play scraping
- `analyzer.py`: BERT + TextBlob sentiment
- `topic_modeler.py`: LDA & NMF
- `issue_detector.py`: Issue ranking