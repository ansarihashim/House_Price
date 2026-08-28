"""Streamlit frontend for the California House Price Prediction model.

Pure UI layer — all preprocessing and inference logic lives in model/,
unchanged. This file only collects input, calls predict_house_price(),
and renders the result.
"""

import streamlit as st

from model.inference import predict_house_price

GITHUB_URL = "https://github.com/ansarihashim/House_Price"

st.set_page_config(
    page_title="California Housing — Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #0A0A0E;
        --card-bg: #101016;
        --card-bg-raised: #131319;
        --border: rgba(255,255,255,0.08);
        --border-strong: rgba(255,255,255,0.14);
        --text-primary: #F5F5F7;
        --text-secondary: #A3A3AC;
        --text-tertiary: #6B6B73;
        --accent: #4A4490;
        --accent-hover: #5C54AA;
        --accent-soft: #A79FDE;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    html, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        scroll-behavior: smooth;
    }
    [data-testid="stMarkdownContainer"] p { line-height: 1.6; }

    /* Chrome cleanup */
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    [data-testid="stToolbar"] { right: 1rem; }
    [data-testid="collapsedControl"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    [data-testid="stAppViewContainer"] { background: var(--bg); }
    [data-testid="stMainBlockContainer"] {
        max-width: 940px;
        padding-top: 0.5rem;
        padding-bottom: 4rem;
    }

    .section-anchor { scroll-margin-top: 5.5rem; }

    /* Top navigation — minimal, sticky, quiet */
    .topnav {
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(10,10,14,0.88);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid var(--border);
        margin: -0.5rem -1rem 0;
    }
    .nav-inner {
        max-width: 940px;
        margin: 0 auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.15rem 1rem;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--text-primary);
        font-size: 1rem;
    }
    .brand-mark {
        width: 9px;
        height: 9px;
        border-radius: 3px;
        background: linear-gradient(135deg, var(--accent-soft), var(--accent));
        flex-shrink: 0;
    }
    .nav-links a {
        color: var(--text-secondary) !important;
        text-decoration: none !important;
        margin-left: 2.2rem;
        font-size: 0.88rem;
        font-weight: 500;
        transition: color 0.15s ease;
    }
    .nav-links a:hover { color: var(--text-primary) !important; }

    /* Hero — large, calm, generous whitespace */
    .hero {
        position: relative;
        text-align: center;
        padding: 7rem 1rem 4.5rem;
        overflow: hidden;
    }
    .hero-glow {
        position: absolute;
        top: -14%;
        left: 50%;
        transform: translateX(-50%);
        width: 620px;
        height: 620px;
        background: radial-gradient(circle, rgba(74,68,144,0.26) 0%, rgba(45,42,90,0.10) 45%, rgba(10,10,14,0) 72%);
        z-index: 0;
        pointer-events: none;
    }
    .eyebrow {
        position: relative;
        z-index: 1;
        color: var(--accent-soft);
        letter-spacing: 0.16em;
        font-size: 0.74rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 1.3rem;
    }
    .hero-title {
        position: relative;
        z-index: 1;
        font-size: clamp(2.1rem, 4.4vw, 3.15rem);
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--text-primary);
        margin: 0 0 1.25rem;
        line-height: 1.18;
    }
    .hero-sub {
        position: relative;
        z-index: 1;
        max-width: 560px;
        margin: 0 auto 2.4rem;
        color: var(--text-secondary);
        font-size: 1.02rem;
        line-height: 1.65;
    }
    .cta-button {
        position: relative;
        z-index: 1;
        display: inline-block;
        background: var(--accent);
        color: var(--text-primary) !important;
        text-decoration: none !important;
        padding: 0.85rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.96rem;
        border: 1px solid var(--border-strong);
        box-shadow: 0 6px 20px rgba(74,68,144,0.28);
        transition: background 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
    }
    .cta-button:hover {
        background: var(--accent-hover);
        transform: translateY(-1px);
        box-shadow: 0 8px 24px rgba(74,68,144,0.38);
    }

    /* Cards (bordered containers, targeted via key -> st-key-<name>) */
    .st-key-predictor-card,
    .st-key-result-card,
    .st-key-model-card {
        background: var(--card-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 2.5rem 2.75rem !important;
        margin-top: 1rem !important;
    }
    .st-key-result-card {
        border-top: 2px solid rgba(167,159,222,0.55) !important;
        margin-top: 2rem !important;
        padding: 2.75rem !important;
    }
    .card-title {
        color: var(--text-primary);
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin: 0 0 0.35rem;
    }
    .card-subtitle {
        color: var(--text-secondary);
        font-size: 0.94rem;
        line-height: 1.55;
        margin: 0 0 2rem;
    }

    /* Form fields — consistent, quiet, clear */
    .st-key-predictor-card [data-testid="stVerticalBlock"] {
        gap: 1.4rem;
    }
    [data-testid="stWidgetLabel"] div[data-testid="stMarkdownContainer"] p {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stTooltipIcon"] svg { color: var(--text-tertiary) !important; }
    [data-testid="stNumberInputContainer"] {
        border-radius: 10px !important;
        transition: box-shadow 0.15s ease;
    }
    [data-testid="stNumberInputContainer"]:focus-within {
        box-shadow: 0 0 0 1px var(--accent-soft);
        border-radius: 10px;
    }

    /* Predict button spacing */
    div[data-testid="stButton"] { margin-top: 0.75rem; }

    /* Primary button */
    div[data-testid="stButton"] button[kind="primary"] {
        background: var(--accent);
        border: none;
        border-radius: 10px;
        padding: 0.85rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.005em;
        box-shadow: 0 4px 16px rgba(74,68,144,0.30);
        transition: background 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: var(--accent-hover);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(74,68,144,0.4);
    }

    /* Result card — the dollar figure is the focal point, everything else is quiet */
    .result-heading {
        color: var(--text-secondary);
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
    }
    .result-figure {
        color: var(--text-primary);
        font-size: clamp(2.6rem, 6vw, 3.8rem);
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 0.5rem;
    }
    .result-caption {
        color: var(--accent-soft);
        font-size: 0.92rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .result-secondary {
        color: var(--text-tertiary);
        font-size: 0.84rem;
        margin-bottom: 0.6rem;
    }
    .result-disclaimer {
        color: var(--text-tertiary);
        font-size: 0.78rem;
        line-height: 1.5;
        border-top: 1px solid var(--border);
        padding-top: 1.1rem;
        margin-top: 1rem;
    }

    /* Model stat tiles */
    .stat-tile {
        background: rgba(255,255,255,0.025);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.15rem 0.75rem;
        text-align: center;
    }
    .stat-label {
        color: var(--text-secondary);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    .stat-value {
        color: var(--text-primary);
        font-size: 1.4rem;
        font-weight: 700;
    }
    .stat-sub {
        color: var(--text-tertiary);
        font-size: 0.71rem;
        margin-top: 0.35rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 3.5rem 1rem 1.5rem;
        border-top: 1px solid var(--border);
        margin-top: 3.5rem;
    }
    .footer-title {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 0.92rem;
        margin-bottom: 0.35rem;
    }
    .footer-sub {
        color: var(--text-tertiary);
        font-size: 0.8rem;
        margin-bottom: 1rem;
    }
    .footer-link {
        color: var(--accent-soft) !important;
        text-decoration: none !important;
        font-size: 0.86rem;
        font-weight: 600;
    }
    .footer-link:hover { color: var(--text-primary) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Top navigation
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="topnav">
      <div class="nav-inner">
        <div class="brand"><span class="brand-mark"></span>California Housing</div>
        <div class="nav-links">
          <a href="#predictor">Predictor</a>
          <a href="#model">Model</a>
          <a href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer">GitHub</a>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-glow"></div>
      <p class="eyebrow">CALIFORNIA HOUSING • MACHINE LEARNING</p>
      <h1 class="hero-title">Predict California House Values</h1>
      <p class="hero-sub">
        Estimate the median house value of a California census block group
        from demographic and geographic characteristics.
      </p>
      <a href="#predictor" class="cta-button">Start Prediction</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Prediction section
# --------------------------------------------------------------------------
st.markdown('<div id="predictor" class="section-anchor"></div>', unsafe_allow_html=True)

with st.container(border=True, key="predictor-card"):
    st.markdown('<div class="card-title">Household & Location Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Enter the characteristics of the district.</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        med_inc = st.number_input(
            "Median Income",
            min_value=0.5,
            max_value=15.0,
            value=3.5,
            step=0.1,
            format="%.1f",
            help="Median income in tens of thousands of dollars.",
        )
        ave_rooms = st.number_input(
            "Average Rooms",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.1,
            format="%.1f",
            help="Average number of rooms per household.",
        )
        population = st.number_input(
            "Population",
            min_value=1,
            max_value=40000,
            value=1000,
            step=50,
            help="Total population of the district.",
        )
        latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=42.0,
            value=35.0,
            step=0.01,
            format="%.2f",
            help="Geographic latitude of the district.",
        )

    with col_right:
        house_age = st.number_input(
            "House Age",
            min_value=1,
            max_value=52,
            value=25,
            step=1,
            help="Median age of houses in the district.",
        )
        ave_bedrms = st.number_input(
            "Average Bedrooms",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.1,
            format="%.1f",
            help="Average number of bedrooms per household.",
        )
        ave_occup = st.number_input(
            "Average Occupancy",
            min_value=0.5,
            max_value=20.0,
            value=3.0,
            step=0.1,
            format="%.1f",
            help="Average number of household members.",
        )
        longitude = st.number_input(
            "Longitude",
            min_value=-125.0,
            max_value=-114.0,
            value=-119.0,
            step=0.01,
            format="%.2f",
            help="Geographic longitude of the district.",
        )

    st.write("")
    predict_clicked = st.button(
        "Predict House Value →",
        type="primary",
        use_container_width=True,
        key="predict-btn",
    )

if predict_clicked:
    try:
        st.session_state["prediction"] = predict_house_price(
            MedInc=med_inc,
            HouseAge=house_age,
            AveRooms=ave_rooms,
            AveBedrms=ave_bedrms,
            Population=population,
            AveOccup=ave_occup,
            Latitude=latitude,
            Longitude=longitude,
        )
        st.session_state["prediction_error"] = None
    except (ValueError, TypeError) as exc:
        st.session_state["prediction"] = None
        st.session_state["prediction_error"] = str(exc)

if st.session_state.get("prediction_error"):
    st.error(f"Invalid input: {st.session_state['prediction_error']}")

if st.session_state.get("prediction") is not None:
    prediction = st.session_state["prediction"]
    dollars = round(prediction * 100_000 / 1000) * 1000

    with st.container(border=True, key="result-card"):
        st.markdown(
            f"""
            <div class="result-heading">Estimated Median House Value</div>
            <div class="result-figure">${dollars:,.0f}</div>
            <div class="result-caption">Model prediction</div>
            <div class="result-secondary">Model output: {prediction:.2f} ($100,000s)</div>
            <div class="result-disclaimer">
                This is a machine-learning estimate based on the supplied district characteristics.
            </div>
            """,
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------
# Model section
# --------------------------------------------------------------------------
st.markdown('<div id="model" class="section-anchor"></div>', unsafe_allow_html=True)
st.write("")

with st.container(border=True, key="model-card"):
    st.markdown('<div class="card-title">The Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-subtitle">'
        "Gradient Boosting Regressor trained on the California Housing dataset "
        "(20,640 records, 12 features after engineering)."
        "</div>",
        unsafe_allow_html=True,
    )

    stat_cols = st.columns(4)
    stats = [
        ("MAE", "0.3300", "≈ $33,000 avg. error"),
        ("RMSE", "0.4899", "Held-out test set"),
        ("R²", "0.8168", "Variance explained"),
        ("CV MAE", "0.3282", "± 0.0051, 5-fold"),
    ]
    for col, (label, value, sub) in zip(stat_cols, stats):
        with col:
            st.markdown(
                f"""
                <div class="stat-tile">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}</div>
                    <div class="stat-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="footer">
      <div class="footer-title">California House Price Prediction</div>
      <div class="footer-sub">Built with Python • Scikit-learn • Streamlit</div>
      <a class="footer-link" href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer">GitHub →</a>
    </div>
    """,
    unsafe_allow_html=True,
)
