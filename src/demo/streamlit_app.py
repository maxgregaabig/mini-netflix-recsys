"""Streamlit demo client for the Mini-Netflix Recommender."""
from __future__ import annotations

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Mini-Netflix Recommender", layout="wide")
st.title("Mini-Netflix Recommender")


@st.cache_data(ttl=60)
def fetch_recs(user_id: int, seed_item_id: int | None = None) -> dict | None:
    params = {"user_id": user_id}
    if seed_item_id is not None:
        params["seed_item_id"] = seed_item_id
    try:
        resp = requests.get(f"{API_URL}/recs/home", params=params, timeout=5)
        if resp.status_code == 200:
            return resp.json()
        st.error(f"API returned {resp.status_code}: {resp.text}")
        return None
    except requests.ConnectionError:
        st.error("Cannot connect to API. Make sure the server is running on localhost:8000")
        return None


def check_health() -> bool:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


# Sidebar
with st.sidebar:
    st.header("Settings")
    healthy = check_health()
    st.markdown(f"API Status: {'**Online**' if healthy else '**Offline**'}")

    user_id = st.number_input("User ID", min_value=1, max_value=1000, value=1)
    seed_item = st.number_input("Seed Item ID (for 'Because You Watched')", min_value=0, value=0)
    seed_item_id = seed_item if seed_item > 0 else None

    if st.button("Get Recommendations"):
        st.session_state["fetch"] = True

# Main content
if st.session_state.get("fetch") or healthy:
    data = fetch_recs(user_id, seed_item_id)

    if data:
        st.markdown(f"*Latency: {data.get('latency_ms', '?')} ms*")

        for row in data.get("rows", []):
            row_name = row["row_name"].replace("_", " ").title()
            st.subheader(row_name)

            items = row.get("items", [])
            if not items:
                st.info("No items in this row.")
                continue

            is_trending = row["row_name"] == "trending"
            cols = st.columns(min(len(items), 5))
            for i, item in enumerate(items):
                with cols[i % len(cols)]:
                    genres_str = ", ".join(item.get("genres", []))
                    score_label = f"{item['score']:.0f} watches" if is_trending else f"{item['score']:.0f}% Match"
                    st.markdown(f"""
**{item.get('title', f'Item {item["item_id"]}')}**

{score_label}

{genres_str}
                    """)

            st.divider()

# Log event form
with st.expander("Log an Event"):
    col1, col2 = st.columns(2)
    with col1:
        ev_user = st.number_input("Event User ID", min_value=1, value=1, key="ev_user")
        ev_item = st.number_input("Event Item ID", min_value=1, value=1, key="ev_item")
    with col2:
        ev_type = st.selectbox("Event Type", ["watch", "skip", "click"])
        ev_rating = st.slider("Rating", 0.5, 5.0, 4.0, 0.5)

    if st.button("Submit Event"):
        try:
            resp = requests.post(f"{API_URL}/log_event", json={
                "user_id": ev_user,
                "item_id": ev_item,
                "event_type": ev_type,
                "rating": ev_rating,
            }, timeout=5)
            if resp.status_code == 200:
                st.success("Event logged!")
            else:
                st.error(f"Error: {resp.text}")
        except requests.ConnectionError:
            st.error("Cannot connect to API")
