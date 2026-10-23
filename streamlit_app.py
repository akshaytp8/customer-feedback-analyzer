"""
Streamlit frontend for shashwat-k19/customer-feedback-analyzer.

The existing Flask backend and ML pipeline remain unchanged.
Run the backend separately, then run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import json
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(
    page_title="FeedbackLens | Customer Feedback Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = os.getenv("FEEDBACK_API_URL", "http://127.0.0.1:5001/api").rstrip("/")
REQUEST_TIMEOUT = (10, 300)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.8rem; padding-bottom: 2.5rem; }
      [data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.18);
        padding: 14px 16px; border-radius: 12px;
      }
      .small-muted { opacity: .75; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_request(method: str, endpoint: str, *, base_url: str, payload: dict | None = None) -> dict[str, Any]:
    """Call the existing Flask API and return a JSON object."""
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        response = requests.request(
            method=method,
            url=url,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "The backend request timed out. Analysis of a large number of reviews can take several minutes."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot connect to the backend at {base_url}. Start the Flask server and verify the API URL."
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"API request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"The backend returned a non-JSON response (HTTP {response.status_code}).") from exc

    if not response.ok or (isinstance(data, dict) and data.get("success") is False):
        message = data.get("error") or data.get("message") or f"HTTP {response.status_code}"
        raise RuntimeError(str(message))

    if not isinstance(data, dict):
        raise RuntimeError("Unexpected response from backend: expected a JSON object.")
    return data


def as_number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        return result if pd.notna(result) else default
    except (TypeError, ValueError):
        return default


def records_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, list):
        return pd.DataFrame(value)
    return pd.DataFrame()


def get_app_label(app: dict) -> str:
    title = app.get("title") or app.get("name") or "Unknown app"
    developer = app.get("developer")
    app_id = app.get("app_id") or app.get("appId") or ""
    suffix = f" — {developer}" if developer else ""
    return f"{title}{suffix} ({app_id})" if app_id else f"{title}{suffix}"


def render_sentiment(analysis: dict) -> None:
    details = analysis.get("app_details") or {}
    sentiment = analysis.get("sentiment_distribution") or {}
    insights = analysis.get("insights") or {}
    total = int(analysis.get("total_reviews") or 0)
    positive = int(sentiment.get("positive") or 0)
    neutral = int(sentiment.get("neutral") or 0)
    negative = int(sentiment.get("negative") or 0)
    avg_score = as_number(sentiment.get("avg_score"))
    health = as_number(insights.get("health_score"))
    positive_pct = positive / total * 100 if total else 0

    st.markdown(f"## {details.get('title', 'App analysis')}")
    metadata = []
    if details.get("developer"):
        metadata.append(str(details["developer"]))
    if details.get("category"):
        metadata.append(str(details["category"]))
    if details.get("rating") is not None:
        metadata.append(f"Store rating: {details['rating']}")
    if metadata:
        st.caption(" · ".join(metadata))
    if analysis.get("from_cache"):
        st.caption("Loaded from the backend's analysis cache.")

    cols = st.columns(4)
    cols[0].metric("Reviews analyzed", f"{total:,}")
    cols[1].metric("Positive reviews", f"{positive:,}", f"{positive_pct:.1f}% of total")
    cols[2].metric("Average sentiment", f"{avg_score:.2f}", "Scale depends on backend model")
    cols[3].metric("Health score", f"{health:.1f}/100", insights.get("health_status", ""))

    st.markdown("### Sentiment overview")
    sentiment_df = pd.DataFrame(
        {"Sentiment": ["Positive", "Neutral", "Negative"], "Reviews": [positive, neutral, negative]}
    )
    chart_col, detail_col = st.columns([1.2, 1])
    with chart_col:
        if total:
            fig = px.pie(
                sentiment_df,
                names="Sentiment",
                values="Reviews",
                hole=0.55,
                title="Review sentiment distribution",
            )
            fig.update_layout(margin=dict(t=55, b=10, l=10, r=10), legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sentiment counts were returned.")
    with detail_col:
        st.dataframe(sentiment_df, hide_index=True, use_container_width=True)
        st.caption("Sentiment labels are model predictions, not verified ground truth.")

    ratings = records_frame(analysis.get("rating_distribution"))
    if not ratings.empty:
        st.markdown("### Star rating distribution")
        st.plotly_chart(px.bar(ratings, x="rating", y="count", title="Store ratings", labels={"rating": "Star rating", "count": "Reviews"}), use_container_width=True)


def render_topics(analysis: dict) -> None:
    st.subheader("Topic discovery")
    left, right = st.columns(2)
    for container, key, label in (
        (left, "topics_lda", "LDA topics"),
        (right, "topics_nmf", "NMF topics"),
    ):
        with container:
            st.markdown(f"**{label}**")
            frame = records_frame(analysis.get(key))
            if frame.empty:
                st.info("No topic results returned.")
            else:
                st.dataframe(frame, hide_index=True, use_container_width=True)


def render_issues(analysis: dict) -> None:
    st.subheader("Prioritized issues")
    issues = records_frame(analysis.get("top_issues"))
    if issues.empty:
        st.info("No issue-priority records were returned.")
    else:
        display = issues.copy()
        for col in ("priority_score", "percentage", "mentions"):
            if col in display.columns:
                display[col] = pd.to_numeric(display[col], errors="coerce")
        sort_col = "priority_score" if "priority_score" in display.columns else None
        if sort_col:
            display = display.sort_values(sort_col, ascending=False)
        st.dataframe(display, hide_index=True, use_container_width=True)

    trending = records_frame(analysis.get("trending_issues"))
    if not trending.empty:
        st.markdown("**Trending issues**")
        st.dataframe(trending, hide_index=True, use_container_width=True)


def render_insights(analysis: dict) -> None:
    insights = analysis.get("insights") or {}
    st.subheader("Actionable insights")
    priority = insights.get("top_priority") or {}
    if priority:
        st.markdown("#### Highest-priority issue")
        st.write(f"**Issue:** {priority.get('issue', 'Not available')}")
        st.write(f"**Severity:** {priority.get('severity', 'Not available')}")
        st.write(f"**Affected reviews:** {priority.get('affected_users', 'Not available')}")
        st.write(priority.get("recommendation", "No recommendation returned."))
        samples = priority.get("sample_reviews")
        if isinstance(samples, list) and samples:
            with st.expander("Example reviews for this issue"):
                for item in samples[:10]:
                    if isinstance(item, dict):
                        st.write(item.get("content") or item.get("review") or str(item))
                    else:
                        st.write(str(item))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Strengths")
        strengths = insights.get("strengths") or []
        if strengths:
            for item in strengths:
                if isinstance(item, dict):
                    st.write(f"• **{item.get('attribute', 'Strength')}** — {item.get('mentions', 'N/A')} mentions ({item.get('percentage', 'N/A')}%)")
                else:
                    st.write(f"• {item}")
        else:
            st.caption("No strengths were identified by the backend.")
    with right:
        st.markdown("#### Weaknesses")
        weaknesses = insights.get("weaknesses") or []
        if weaknesses:
            for item in weaknesses:
                if isinstance(item, dict):
                    st.write(f"• **{item.get('issue', 'Issue')}** — severity {item.get('severity', 'N/A')}, affected {item.get('affected_users', 'N/A')}")
                else:
                    st.write(f"• {item}")
        else:
            st.caption("No weaknesses were identified by the backend.")

    focus = insights.get("topic_focus_areas") or {}
    if isinstance(focus, dict) and focus:
        st.markdown("#### Topic focus areas")
        focus_df = pd.DataFrame(
            [
                {"Topic": topic, **(values if isinstance(values, dict) else {"value": values})}
                for topic, values in focus.items()
            ]
        )
        st.dataframe(focus_df, hide_index=True, use_container_width=True)

    trend = insights.get("sentiment_trend") or {}
    if trend:
        st.markdown("#### Sentiment trend")
        direction = str(trend.get("direction", "not available")).replace("_", " ").title()
        st.write(f"**Trend:** {direction}")
        if trend.get("change_percent") is not None:
            st.write(f"Change: {trend.get('change_percent')} (backend-reported units)")


def render_recent_reviews(analysis: dict) -> None:
    st.subheader("Recent reviews")
    reviews = records_frame(analysis.get("recent_reviews"))
    if reviews.empty:
        st.info("No recent reviews were returned.")
        return
    st.dataframe(reviews, hide_index=True, use_container_width=True)
    st.download_button(
        "Download recent reviews as CSV",
        data=reviews.to_csv(index=False).encode("utf-8"),
        file_name="recent_reviews.csv",
        mime="text/csv",
    )


st.title("📊 FeedbackLens")
st.caption("Customer Feedback Analyzer · Streamlit frontend for the existing Flask + ML backend")

with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("Backend API URL", value=DEFAULT_API_URL, help="Example: http://127.0.0.1:5001/api")
    if st.button("Check backend connection", use_container_width=True):
        try:
            health = api_request("GET", "/health", base_url=api_url)
            st.success(f"Backend status: {health.get('status', 'reachable')}")
        except RuntimeError as exc:
            st.error(str(exc))

    st.divider()
    st.markdown("### Analysis settings")
    review_count = st.slider("Number of reviews", min_value=50, max_value=2000, value=500, step=50)
    force_refresh = st.checkbox("Force fresh analysis", value=False, help="Ignore cached results and scrape/analyze reviews again.")
    st.caption("Fresh analysis may take several minutes, depending on the backend and selected review count.")

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "selected_app" not in st.session_state:
    st.session_state.selected_app = None

search_col, search_button_col = st.columns([4, 1])
with search_col:
    query = st.text_input("Search Google Play Store apps", placeholder="Try Instagram, WhatsApp, Spotify…")
with search_button_col:
    st.write("")
    st.write("")
    search_clicked = st.button("Search apps", type="primary", use_container_width=True)

if search_clicked:
    if len(query.strip()) < 2:
        st.warning("Enter at least two characters to search.")
    else:
        with st.spinner("Searching Google Play Store…"):
            try:
                result = api_request("POST", "/search", base_url=api_url, payload={"query": query.strip()})
                apps = result.get("apps") or []
                if not apps:
                    st.warning("No apps were found. Try a different search.")
                    st.session_state.search_results = []
                else:
                    st.session_state.search_results = apps
            except RuntimeError as exc:
                st.error(str(exc))

apps = st.session_state.get("search_results", [])
if apps:
    labels = {get_app_label(app): app for app in apps}
    selected_label = st.selectbox("Select an app", list(labels.keys()))
    selected_app = labels[selected_label]
    st.session_state.selected_app = selected_app

    app_cols = st.columns([1, 3])
    with app_cols[0]:
        icon_url = selected_app.get("icon")
        if icon_url:
            st.image(icon_url, width=100)
    with app_cols[1]:
        st.markdown(f"### {selected_app.get('title', 'Selected app')}")
        st.write(f"Developer: {selected_app.get('developer', 'Unknown')}")
        st.write(f"App ID: `{selected_app.get('app_id', '')}`")
        if selected_app.get("score") is not None:
            st.write(f"Store score: {selected_app['score']}")

    if st.button("🚀 Start analysis", type="primary", use_container_width=True):
        app_id = selected_app.get("app_id")
        if not app_id:
            st.error("The selected app does not include an app ID.")
        else:
            progress = st.progress(0, text="Starting analysis…")
            try:
                progress.progress(10, text="Requesting review analysis…")
                result = api_request(
                    "POST",
                    "/analyze",
                    base_url=api_url,
                    payload={
                        "app_id": app_id,
                        "review_count": int(review_count),
                        "force_refresh": bool(force_refresh),
                    },
                )
                progress.progress(100, text="Analysis complete.")
                st.session_state.analysis_result = result
                st.session_state.analysis_app_id = app_id
            except RuntimeError as exc:
                st.error(str(exc))
            finally:
                progress.empty()

analysis = st.session_state.get("analysis_result")
if analysis:
    st.divider()
    render_sentiment(analysis)
    tabs = st.tabs(["Insights", "Topics", "Issues", "Recent reviews", "Export"])
    with tabs[0]:
        render_insights(analysis)
    with tabs[1]:
        render_topics(analysis)
    with tabs[2]:
        render_issues(analysis)
    with tabs[3]:
        render_recent_reviews(analysis)
    with tabs[4]:
        st.subheader("Export analysis")
        st.write("Download the full analysis response as JSON or export the issue/review tables as CSV.")
        st.download_button(
            "Download full analysis as JSON",
            data=json.dumps(analysis, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
            file_name="feedback_analysis.json",
            mime="application/json",
        )
        export_cols = st.columns(2)
        issues_df = records_frame(analysis.get("top_issues"))
        reviews_df = records_frame(analysis.get("recent_reviews"))
        with export_cols[0]:
            if not issues_df.empty:
                st.download_button(
                    "Download prioritized issues CSV",
                    data=issues_df.to_csv(index=False).encode("utf-8"),
                    file_name="prioritized_issues.csv",
                    mime="text/csv",
                )
        with export_cols[1]:
            if not reviews_df.empty:
                st.download_button(
                    "Download recent reviews CSV",
                    data=reviews_df.to_csv(index=False).encode("utf-8"),
                    file_name="recent_reviews.csv",
                    mime="text/csv",
                )
else:
    st.info("Search for an app, select it, and click **Start analysis** to view the dashboard.")

st.divider()
st.caption("Frontend adaptation only. Review scraping, sentiment analysis, topic modeling, issue detection and caching are handled by the original Flask backend.")
