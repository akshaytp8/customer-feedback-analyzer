# Customer Feedback Analyzer

An AI-powered dashboard for analyzing Google Play Store reviews, identifying sentiment and topics, and surfacing customer issues. This modified version uses a **Streamlit frontend** with the project's existing **Flask API and ML/NLP backend**.

## Features

- Search for apps and retrieve Google Play Store reviews through the backend
- Analyze review sentiment
- Explore topics and recurring issues
- View visual summaries and actionable insights
- Use a Streamlit interface instead of the original React interface

## Tech Stack

- **Frontend:** Python, Streamlit, Pandas, Plotly, Requests
- **Backend:** Flask, Python
- **ML/NLP:** Hugging Face Transformers, TextBlob, NLTK, scikit-learn
- **Data:** Pandas, NumPy, cached analysis results

## Project Structure

The tree below shows the original repository areas and the Streamlit files added during the frontend conversion. Some generated folders, cache contents, and model files are omitted.

```text
customer-feedback-analyzer-main/
├── backend/
│   ├── app.py
│   ├── scraper.py
│   ├── preprocessing.py
│   ├── sentiment_analyzer.py
│   ├── topic_modeler.py
│   ├── issue_detector.py
│   ├── config.py
│   ├── utils.py
│   ├── requirements.txt
│   └── cache/
├── docs/
│   ├── architecture.md
│   └── api_documentation.md
├── tests/
│   ├── test_scraper.py
│   ├── test_sentiment.py
│   └── test_topics.py
├── .streamlit/
│   └── config.toml
├── streamlit_app.py
├── requirements.txt
├── .gitignore
├── LICENSE
├── README.md
└── setup.sh
```

**Legacy frontend files:** If you have not deleted the old React frontend yet, its `frontend/` folder and Node package files may still be present. They are not required to run the Streamlit interface. Keep or remove them based on whether you want to retain the React version.

The `.venv/` folder is created locally during setup and should not be committed to Git.

## Requirements

- Python 3.11 recommended for the current dependency versions
- Internet connection for package installation and initial model download
- Git, if cloning the repository

## Installation on Windows

Run these commands from the project root.

### 1. Create and activate a virtual environment

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. Install backend dependencies

```bat
python -m pip install -r backend\requirements.txt
```

### 3. Install Streamlit frontend dependencies

```bat
python -m pip install -r requirements.txt
```

## Run the Application

Use two terminals. Activate `.venv` in each terminal.

### Terminal 1: Start Flask

```bat
cd backend
python app.py
```

In your current setup, Flask listens on `http://127.0.0.1:5001`. Leave this terminal running.

### Terminal 2: Start Streamlit

Return to the project root, then run:

```bat
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

In the Streamlit sidebar, set the backend API URL to:

```text
http://127.0.0.1:5001/api
```

Use the actual port and API prefix configured by your Flask backend if they differ.

## Troubleshooting

- **Cannot connect to backend:** Check that Flask is running and that the Streamlit API URL points to the correct port.
- **Missing packages:** Confirm `.venv` is activated, then install both requirements files.
- **Model download failure:** Check the internet connection; the NLP model may download on the first backend run.
- **NumPy/PyArrow compatibility error:** Use compatible versions in the root `requirements.txt` and install it in the same environment used to run Streamlit.

## Tests

The repository includes backend tests under `tests/`. Run them from the project root using the active virtual environment:

```bat
python -m pip install pytest
python -m pytest tests -v
```

If a test depends on network access or external model downloads, its behavior may vary with connectivity and local configuration.

## Notes

- The Flask backend must be running for the Streamlit frontend to use its API.
- Keep `backend/`, `docs/`, `tests/`, `.gitignore`, and `LICENSE`.
- Do not commit `.venv/`, API secrets, or unnecessary generated cache files.
- Check the original repository's license before redistributing or publishing a modified version.
