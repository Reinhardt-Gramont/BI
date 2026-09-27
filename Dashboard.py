"""
Adidas Foot Print — Sales Intelligence Platform
------------------------------------------------
A Streamlit app that reproduces:
  1) The marketing landing page (hero + team section)
  2) The internal BI dashboard you land on after clicking "Get Started":
       - "Overview"  -> Executive overview (KPIs, revenue chart, traffic donut)
       - "Analytics" -> Revenue/Traffic + Business performance + Top products

Dashboard figures are computed from the workspace workbook or a CSV/Excel
dataset uploaded in Settings. Built-in demo values are used only when no
dataset is available. No external live data feed is configured.

Run with:
    streamlit run Dashboard.py
"""

import base64
import json
import re
from io import BytesIO
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pathlib import Path

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Adidas Foot Print — Sales Intelligence Platform",
    page_icon="👣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# COLORS
# ----------------------------------------------------------------------
GREEN = "#a3e635"
GREEN_DARK = "#84cc16"
BLUE = "#60a5fa"
TEAL = "#2dd4bf"
ORANGE = "#f59e0b"
RED = "#f87171"
LAVENDER = "#d7d4fb"
BLACK = "#0a0a0a"
PAGE_BG = "#0b0b0d"
CARD_BG = "#121214"
CARD_BORDER = "#232326"
SIDEBAR_BG = "#0d0d0f"
MUTED = "#9ca3af"
MUTED2 = "#6b7280"

# ----------------------------------------------------------------------
# SESSION STATE / ROUTING
# ----------------------------------------------------------------------
st.session_state.setdefault("page", "landing")   # "landing" | "dashboard"
st.session_state.setdefault("dash_tab", "Overview")  # active sidebar nav item
st.session_state.setdefault("scenario", "Base case")  # simulation scenario toggle
st.session_state.setdefault("sidebar_visible", True)
st.session_state.setdefault("settings_alerts", False)
st.session_state.setdefault("dashboard_region", "All regions")
st.session_state.setdefault("dashboard_channel", "All channels")
st.session_state.setdefault("dashboard_product", "All products")
st.session_state.setdefault("dashboard_retailer", "All retailers")
st.session_state.setdefault("dashboard_state", "All states")
st.session_state.setdefault("dashboard_city", "All cities")
st.session_state.setdefault("dashboard_light_mode", False)
DATE_RANGE_OPTIONS = ["Last 30 days", "Last 90 days", "Last 12 months", "YTD", "All time"]
DATE_RANGE_SECTIONS = ("Overview", "Analytics")
st.session_state.setdefault("section_date_ranges", {})
st.session_state.setdefault("active_dataset", None)


def go_dashboard():
    st.session_state.page = "dashboard"
    st.session_state.dash_tab = "Overview"


def go_landing():
    st.session_state.page = "landing"


def toggle_sidebar():
    st.session_state.sidebar_visible = not st.session_state.sidebar_visible


def set_tab(tab):
    st.session_state.dash_tab = tab


def set_section_date_range(section):
    key = f"date-range-{section.lower()}"
    st.session_state.section_date_ranges[section] = st.session_state[key]


def persist_uploaded_dataset():
    uploaded = st.session_state.get("dataset_upload")
    st.session_state.active_dataset = (
        (uploaded.name, uploaded.getvalue()) if uploaded is not None else None
    )


def clear_uploaded_dataset():
    st.session_state.active_dataset = None
    if "dataset_upload" in st.session_state:
        st.session_state.dataset_upload = None


def render_section_date_range(section):
    key = f"date-range-{section.lower()}"
    if key not in st.session_state:
        st.session_state[key] = st.session_state.section_date_ranges.get(
            section, "Last 12 months"
        )
    return st.selectbox(
        "Date range",
        DATE_RANGE_OPTIONS,
        key=key,
        on_change=set_section_date_range,
        args=(section,),
    )


def set_scenario(scenario):
    st.session_state.scenario = scenario


def toggle_dashboard_theme():
    st.session_state.dashboard_light_mode = not st.session_state.dashboard_light_mode


def reset_dashboard_filters():
    for section in DATE_RANGE_SECTIONS:
        st.session_state.section_date_ranges[section] = "Last 12 months"
        key = f"date-range-{section.lower()}"
        if key in st.session_state:
            st.session_state[key] = "Last 12 months"
    st.session_state.dashboard_region = "All regions"
    st.session_state.dashboard_channel = "All channels"
    st.session_state.dashboard_product = "All products"
    st.session_state.dashboard_retailer = "All retailers"
    st.session_state.dashboard_state = "All states"
    st.session_state.dashboard_city = "All cities"


# ----------------------------------------------------------------------
# STATIC DEFAULT DATA
# ----------------------------------------------------------------------
DEFAULTS = {
    # landing hero card
    "projected_sales": 1.48, "growth_pct": 15.6,
    "products": 24, "markets": 5, "confidence": 88,
    "bar_values": [3.2, 3.6, 2.9, 4.0, 4.3, 4.1, 4.6, 6.4, 6.9, 6.7, 7.8],
    "bar_months": ["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "bar_highlight_from": 7,

    # KPI row (Executive overview)
    "kpi_revenue": 1.28, "kpi_revenue_delta": 12.5,
    "kpi_profit": 284.6, "kpi_profit_delta": 8.2, "kpi_profit_suffix": "K",
    "kpi_orders": 12486, "kpi_orders_delta": 4.8, "kpi_orders_label": "Total orders",
    "kpi_conv": 3.84, "kpi_conv_delta": -0.6, "kpi_conv_label": "Conversion rate",

    # Revenue overview bar chart (12 months, $K)
    "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "revenue_series": [70, 75, 68, 92, 82, 96, 90, 102, 94, 100, 96, 112],
    "expenses_series": [54, 57, 52, 66, 60, 70, 64, 74, 66, 71, 68, 78],

    # Sales method mix (illustrative demo values)
    "traffic_total": "Demo",
    "traffic_labels": ["Online", "Outlet", "In-store"],
    "traffic_values": [38, 32, 30],
    "traffic_colors": [GREEN, BLUE, TEAL, ORANGE],

    # Business performance line chart ($K, 12 months)
    "perf_revenue": [40, 46, 44, 52, 58, 60, 66, 64, 72, 78, 82, 92],
    "perf_profit": [22, 24, 25, 28, 30, 31, 34, 35, 38, 40, 42, 46],

    # Top products
    "top_products": [
        {"name": "Enterprise Suite", "category": "Software", "value": 142.8, "color": GREEN},
        {"name": "Growth Analytics", "category": "Subscription", "value": 98.4, "color": BLUE},
        {"name": "Data Connect", "category": "Integration", "value": 76.2, "color": TEAL},
        {"name": "Insight Cloud", "category": "Platform", "value": 54.9, "color": ORANGE},
    ],
    "low_products": [
        {"name": "Insight Cloud", "category": "Platform", "value": 54.9, "color": ORANGE},
    ],
    "regional_performance": [],
    "data_rows": 0,
    "date_range": "Demo data",
    "region_count": 5,
}

TEAM = [
    {"name": "Odtojan, Ian Carl", "role": "Lead Analyst", "image": "Odtojan, Ian.jpg"},
    {"name": "Zapanta, Joseph R.", "role": "Data Analyst", "image": "Zapanta, Joseph R..jpg"},
    {"name": "Moreno, Ralph Benedict", "role": "Programmer/Designer", "image": "Moreno, Ralph Benedict Image.jpg"},
    {"name": "Lumandong, Russel", "role": "BI Analyst", "image": "Lumandong, Russel.png"},
]

# ---- Simulation ("Sales forecasting lab") static defaults --------------
SIM_DEFAULTS = {
    "growth_pct": 15.6,
    # Demo history is in $K; the forecast is trained from these values.
    "hist_series": [10, 14, 13, 18, 22, 24, 30],
    "hist_months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"],
    "forecast_months": ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb"],
    "product_outlook": [
        {"name": "Enterprise Suite", "value": 142.8, "status": "Strong momentum",
         "forecast": 24, "color": GREEN, "bar_pct": 100},
        {"name": "Growth Analytics", "value": 98.4, "status": "Above trend",
         "forecast": 18, "color": BLUE, "bar_pct": 77},
        {"name": "Data Connect", "value": 76.2, "status": "Stable demand",
         "forecast": 9, "color": TEAL, "bar_pct": 53},
        {"name": "Insight Cloud", "value": 54.9, "status": "At risk",
         "forecast": -3, "color": ORANGE, "bar_pct": 38},
    ],
    "insight_product": "Enterprise Suite",
    "insight_pct": 41,
}

# Scenario multipliers adjust the trained forecast and its uncertainty band.
SCENARIOS = {
    "Conservative": {"growth_mult": 0.55, "band_mult": 0.65},
    "Base case": {"growth_mult": 1.0, "band_mult": 1.0},
    "Accelerated": {"growth_mult": 1.55, "band_mult": 1.4},
}

NAV_ITEMS = [
    ("Overview", "▦"),
    ("Analytics", "📶"),
    ("Simulation", "✨"),
    ("Products", "🏷️"),
    ("Reports", "📄"),
    ("Settings", "⚙️"),
]

# ----------------------------------------------------------------------
# GLOBAL CSS
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=Inter:wght@400;500;600;700;800;900&display=swap');
    html, body, [class*="css"] {{ font-family:'Inter',sans-serif; }}
    /* Hide the hamburger menu, "made with Streamlit" footer, and Deploy
       button — but keep the header itself (it hosts the sidebar's
       expand/collapse arrow, which must stay usable). */
    #MainMenu {{visibility:hidden;}}
    footer {{visibility:hidden;}}
    div[data-testid="stToolbar"] {{visibility:hidden;}}
    div[data-testid="stDecoration"] {{display:none;}}
    section[data-testid="stSidebar"],
    div[data-testid="stSidebarCollapsedControl"] {{display:none !important;}}
    header[data-testid="stHeader"] {{ background:transparent !important; height:2.6rem; }}
    div[data-testid="stSidebarCollapsedControl"] {{
        color:white !important; top:14px !important; left:10px !important;
    }}
    div[data-testid="stSidebarCollapsedControl"] svg {{ fill:white !important; }}

    .stApp {{
        background: radial-gradient(circle at 85% 5%, rgba(163,230,53,0.06), transparent 40%), {PAGE_BG};
        color:white;
    }}
    .block-container {{ padding-top: 1.2rem; max-width: 1400px; }}

    /* ---- default streamlit button reset (so our scoped rules fully control look) ---- */
    div[data-testid="stButton"] button {{
        background:{CARD_BG} !important; color:#e5e7eb !important;
        border:1px solid {CARD_BORDER} !important; border-radius:9px !important;
        box-shadow:none;
        transition:transform .18s ease, background .18s ease, color .18s ease, filter .18s ease, box-shadow .18s ease;
    }}
    div[data-testid="stPopover"] > button,
    div[data-testid="stPopover"] button {{
        background:{CARD_BG} !important; color:#e5e7eb !important;
        border:1px solid {CARD_BORDER} !important; border-radius:9px !important;
    }}
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    div[data-testid="stTextInput"] input {{
        background:#171719 !important; color:#e5e7eb !important;
        border-color:{CARD_BORDER} !important;
    }}
    div[data-testid="stSelectbox"] [data-baseweb="select"] svg {{ fill:#d1d5db !important; }}
    div[data-testid="stPopoverBody"], div[data-baseweb="popover"],
    div[data-baseweb="menu"], ul[role="listbox"] {{
        background:#151517 !important; color:#e5e7eb !important;
        border-color:{CARD_BORDER} !important;
    }}
    li[role="option"] {{ color:#e5e7eb !important; }}
    li[role="option"]:hover {{ background:#252529 !important; }}
    div[data-testid="stButton"] button:hover {{
        transform:translateY(-1px) scale(1.02) !important;
        filter:brightness(1.12) !important;
        box-shadow:0 8px 18px rgba(255,255,255,0.06) !important;
    }}
    div[data-testid="stButton"] button:active {{
        transform:scale(0.985) !important;
        filter:brightness(1.04) !important;
    }}
    div[data-testid="stButton"] button:focus {{ box-shadow:none; outline:none; }}
    div[data-testid="stButton"] button:focus-visible {{
        outline:2px solid {GREEN} !important; outline-offset:3px !important;
    }}

    /* ---- native widget interaction states ---- */
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    div[data-testid="stFileUploader"] section,
    div[data-testid="stCheckbox"] label,
    div[data-testid="stToggle"] label {{
        transition:transform .18s ease, background .18s ease, border-color .18s ease,
            box-shadow .18s ease, filter .18s ease;
    }}
    div[data-testid="stSelectbox"]:hover [data-baseweb="select"] > div {{
        transform:translateY(-1px); filter:brightness(1.12);
    }}
    div[data-testid="stSelectbox"]:focus-within [data-baseweb="select"] > div {{
        border-color:{GREEN} !important; box-shadow:0 0 0 2px rgba(163,230,53,0.16) !important;
    }}
    div[data-testid="stCheckbox"] label:hover,
    div[data-testid="stToggle"] label:hover {{ filter:brightness(1.18); }}
    div[data-testid="stFileUploader"] section:hover {{
        transform:translateY(-1px); filter:brightness(1.12);
        border-color:{GREEN} !important; box-shadow:0 8px 18px rgba(163,230,53,0.08);
    }}
    div[data-testid="stFileUploader"]:focus-within section {{
        border-color:{GREEN} !important; box-shadow:0 0 0 2px rgba(163,230,53,0.16) !important;
    }}

    /* ---- visual feedback for control-like dashboard affordances ---- */
    .fp-chip, .fp-iconbtn, .view-all {{
        transition:transform .18s ease, filter .18s ease, border-color .18s ease, box-shadow .18s ease;
    }}
    .fp-chip:hover, .fp-iconbtn:hover, .view-all:hover {{
        transform:translateY(-1px) scale(1.025); filter:brightness(1.16);
        border-color:rgba(163,230,53,0.42); box-shadow:0 7px 16px rgba(0,0,0,0.18);
    }}
    @media (prefers-reduced-motion: reduce) {{
        div[data-testid="stButton"] button,
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
        div[data-testid="stFileUploader"] section,
        div[data-testid="stCheckbox"] label,
        div[data-testid="stToggle"] label,
        .fp-chip, .fp-iconbtn, .view-all, .fp-card, .kpi-card, .chart-card,
        .sim-stat-box, .data-scope-item, .prod-row, .reports-row, .outlook-row {{
            transition:none !important;
        }}
    }}

    /* ---- pill buttons: Get Started (landing) + Export report (dashboard) ---- */
    div[class*="st-key-cta-btn"] button, div[class*="st-key-export-btn"] button {{
        background:{GREEN} !important; color:black !important; font-weight:800 !important;
        border-radius:999px !important; padding:10px 24px !important; font-size:14.5px !important;
    }}
    div[class*="st-key-cta-btn"] button:hover, div[class*="st-key-export-btn"] button:hover {{
        background:{GREEN_DARK} !important; color:black !important;
    }}

    /* ---- sidebar nav buttons: simple dark dashboard list ---- */
    div[class*="st-key-navA-"] button, div[class*="st-key-navI-"] button {{
        background:transparent !important; color:#e5e7eb !important; font-weight:600 !important;
        border:none !important; border-radius:10px !important; text-align:left !important; justify-content:flex-start !important;
        padding:10px 10px !important; font-size:15px !important; min-height:42px !important;
        transform:none !important; box-shadow:none !important; margin:2px 0 !important;
        transition:transform .18s ease, background .18s ease, color .18s ease, filter .18s ease, box-shadow .18s ease;
    }}
    div[class*="st-key-navI-"] button {{ color:#d1d5db !important; }}
    div[class*="st-key-navI-"] button:hover {{
        background:rgba(255,255,255,0.035) !important; color:#f3f4f6 !important;
        transform:scale(1.03) !important; filter:brightness(1.12) !important;
        box-shadow:0 8px 18px rgba(255,255,255,0.03) !important;
    }}
    div[class*="st-key-navA-"] button {{
        background:rgba(255,255,255,0.02) !important; color:#f3f4f6 !important;
        transform:scale(1.02) !important; filter:brightness(1.08) !important;
    }}
    div[class*="st-key-navA-settings"] button {{
        background:rgba(163,230,53,0.86) !important; color:#0b0b0d !important; font-weight:800 !important;
        border-radius:10px !important; border:none !important; margin-top:18px !important;
        padding:12px 12px !important; box-shadow:0 10px 26px rgba(163,230,53,0.18) !important;
        transition:transform .18s ease, filter .18s ease, box-shadow .18s ease;
    }}
    div[class*="st-key-navA-settings"] button:hover {{
        transform:scale(1.025) !important; filter:brightness(1.10) !important;
        box-shadow:0 12px 28px rgba(163,230,53,0.24) !important;
    }}
    div[class*="st-key-navA-"] button p, div[class*="st-key-navI-"] button p {{ text-align:left !important; margin:0 !important; }}
    div[class*="st-key-navA-settings"] button p {{ margin:0 !important; }}

    /* ---- back-to-site link button on dashboard pages ---- */
    div[class*="st-key-back-home"] button {{
        background:transparent !important; color:{MUTED} !important; font-weight:600 !important;
        font-size:12.5px !important; padding:2px 0 !important;
    }}
    div[class*="st-key-back-home"] button:hover {{ color:{GREEN} !important; }}
    div[class*="st-key-dashboard-home"] button {{
        background:{CARD_BG} !important; color:#d7dbe4 !important; border:1px solid {CARD_BORDER} !important;
        border-radius:9px !important; font-size:12px !important; font-weight:700 !important;
        padding:7px 12px !important; min-height:34px !important;
    }}
    div[class*="st-key-dashboard-home"] button:hover {{
        background:rgba(163,230,53,0.12) !important; color:{GREEN} !important; border-color:{GREEN} !important;
    }}
    div[class*="st-key-dashboard-home"] button {{ min-width:38px !important; padding:7px 9px !important; }}
    div[class*="st-key-sidebar-toggle"] button {{
        width:44px !important; min-width:44px !important; max-width:44px !important;
        padding:7px 5px !important; min-height:34px !important;
        font-size:12px !important; font-weight:800 !important;
    }}
    div[class*="st-key-theme-toggle"] button,
    div[class*="st-key-notification-trigger"] button {{
        width:44px !important; min-width:44px !important; max-width:44px !important;
        min-height:40px !important; padding:7px !important; flex:0 0 44px !important;
    }}
    div[class*="st-key-dashboard-sidebar-panel"] {{
        background:{SIDEBAR_BG}; border:1px solid {CARD_BORDER}; border-radius:14px;
        box-sizing:border-box; padding:14px 12px; min-height:calc(100vh - 24px);
    }}
    div[data-testid="stColumn"]:has([class*="st-key-dashboard-sidebar-panel"]) {{
        position:sticky !important; top:12px !important; align-self:flex-start !important;
        height:calc(100vh - 24px) !important; overflow-y:auto !important; z-index:20;
    }}
    div[class*="st-key-dashboard-sticky-header"] {{
        position:sticky !important; top:0 !important; align-self:flex-start !important;
        width:100%; box-sizing:border-box; z-index:30;
        margin:0 -8px 8px; padding:8px;
        background:rgba(11,11,13,0.78); backdrop-filter:blur(14px);
        -webkit-backdrop-filter:blur(14px);
        border-bottom:1px solid rgba(156,163,175,0.18);
    }}

    section[data-testid="stSidebar"] {{ background:{SIDEBAR_BG} !important; border-right:1px solid {CARD_BORDER}; }}
    section[data-testid="stSidebar"] .block-container {{ padding-top:1.4rem; }}

    /* ==== TOP BAR ==== */
    .fp-topbar {{
        display:flex; justify-content:space-between; align-items:center;
        padding:14px 0 16px 0; border-bottom:1px solid {CARD_BORDER}; margin-bottom:8px;
    }}
    .fp-logo {{ display:flex; align-items:center; gap:10px; }}
    .fp-logo-sq {{
        width:34px; height:34px; border-radius:10px;
        background:linear-gradient(135deg,{GREEN},{GREEN_DARK});
        display:flex; align-items:center; justify-content:center; font-size:16px;
    }}
    .fp-logo-text b {{ display:block; font-size:14px; font-weight:800; line-height:1.1; }}
    .fp-logo-text span {{ display:block; font-size:10px; color:{MUTED}; letter-spacing:1px; }}
    .fp-badge {{ color:{MUTED}; font-size:12px; font-weight:600; letter-spacing:1.1px; display:flex; align-items:center; gap:8px; }}
    .fp-dot {{ width:7px; height:7px; border-radius:50%; background:{GREEN}; display:inline-block; }}
    .fp-search {{
        background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:12px;
        padding:9px 14px; color:{MUTED}; font-size:13px; display:flex; align-items:center;
        justify-content:space-between; min-width:320px;
    }}
    .fp-livepill {{
        background:rgba(163,230,53,0.12); color:{GREEN}; font-weight:700; font-size:12px;
        padding:6px 14px; border-radius:999px; display:inline-flex; width:max-content;
        max-width:100%; align-items:center; gap:6px; white-space:nowrap;
    }}
    .fp-login-shell {{ max-width:1080px; margin:7vh auto 0; }}
    .fp-login-grid {{ display:grid; grid-template-columns:1.1fr .9fr; gap:44px; align-items:center; }}
    .fp-login-kicker {{ color:{GREEN}; font-size:12px; font-weight:800; letter-spacing:1.5px; margin-bottom:18px; }}
    .fp-login-title {{ font-family:'Archivo Black','Inter',sans-serif; font-size:54px; line-height:1.02; text-transform:uppercase; margin:0 0 20px; }}
    .fp-login-title span {{ color:{GREEN}; }}
    .fp-login-copy {{ color:#b7bbc4; font-size:15px; line-height:1.65; max-width:500px; }}
    .fp-login-card {{ background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:18px; padding:30px; box-shadow:0 20px 80px rgba(0,0,0,.28); }}
    .fp-login-card h2 {{ font-size:23px; margin:0 0 7px; }}
    .fp-login-card p {{ color:{MUTED}; font-size:13px; margin:0 0 24px; }}
    .fp-login-note {{ color:{MUTED2}; font-size:11px; text-align:center; margin-top:16px; }}
    .fp-login-error {{ background:rgba(248,113,113,.12); border:1px solid rgba(248,113,113,.35); color:#fecaca; border-radius:9px; padding:9px 11px; font-size:12px; margin-bottom:13px; }}
    .fp-signed-in {{ color:{MUTED}; font-size:12px; }}
    @media (max-width: 800px) {{ .fp-login-shell {{ margin:2vh auto 0; }} .fp-login-grid {{ grid-template-columns:1fr; gap:25px; }} .fp-login-title {{ font-size:40px; }} }}
    .fp-iconbtn {{
        width:34px; height:34px; border-radius:50%; background:{CARD_BG}; border:1px solid {CARD_BORDER};
        display:flex; align-items:center; justify-content:center; position:relative; font-size:14px;
    }}
    .fp-iconbtn .dot {{ position:absolute; top:5px; right:6px; width:7px; height:7px; border-radius:50%; background:{RED}; }}

    /* ==== SIDEBAR (custom, not Streamlit's) ==== */
    .fp-side {{
        background:{SIDEBAR_BG}; border-right:1px solid {CARD_BORDER};
        padding:18px 14px; border-radius:16px; height:100%;
    }}
    .fp-side-label {{ color:{MUTED2}; font-size:11px; letter-spacing:1.3px; font-weight:700; margin:6px 0 10px 4px; }}

    /* ==== HERO ==== */
    .fp-eyebrow {{ display:flex; align-items:center; gap:8px; color:{GREEN}; font-weight:800; font-size:13px; letter-spacing:1.2px; margin-bottom:18px; }}
    .fp-headline {{ font-family:'Archivo Black','Inter',sans-serif; font-size:56px; line-height:1.05; font-weight:900; text-transform:uppercase; margin:0 0 22px 0; }}
    .fp-headline .accent {{ color:{GREEN}; }}
    .fp-sub {{ color:#c7c7c7; font-size:16px; line-height:1.6; max-width:520px; margin-bottom:18px; }}
    .fp-cta-note {{ color:{MUTED}; font-size:13px; padding-top:8px; display:inline-block; }}

    .fp-card {{ background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:22px; padding:20px 20px 16px 20px; }}
    .fp-card, .kpi-card, .chart-card {{
        transition:transform .18s ease, border-color .18s ease, filter .18s ease,
            box-shadow .18s ease;
    }}
    .fp-card:empty, .chart-card:empty {{ display:none !important; }}
    .fp-card:hover, .kpi-card:hover, .chart-card:hover {{
        transform:translateY(-3px); border-color:rgba(163,230,53,0.42);
        filter:brightness(1.06); box-shadow:0 12px 28px rgba(0,0,0,0.2);
    }}
    .fp-card-label {{ color:{MUTED}; font-size:13px; margin-bottom:6px; }}
    .fp-card-row {{ display:flex; align-items:center; justify-content:space-between; }}
    .fp-card-value {{ font-size:32px; font-weight:900; }}
    .fp-pill {{ background:rgba(163,230,53,0.15); color:{GREEN}; font-weight:800; font-size:13px; padding:6px 12px; border-radius:999px; }}
    .fp-stat-box {{ background:#0d0d0d; border:1px solid {CARD_BORDER}; border-radius:14px; padding:12px 16px; flex:1; }}
    .fp-stat-label {{ color:{MUTED}; font-size:12px; margin-bottom:4px; }}
    .fp-stat-value {{ font-size:20px; font-weight:800; }}

    .fp-section-eyebrow {{ color:{GREEN}; font-weight:800; font-size:12px; letter-spacing:1.5px; margin-bottom:8px; }}
    .fp-section-title {{ font-size:26px; font-weight:800; margin:0; }}
    .fp-section-note {{ color:{MUTED}; font-size:13px; }}
    .fp-team-card {{ background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:14px; padding:14px; display:flex; align-items:center; gap:12px; }}
    .fp-avatar {{ width:46px; height:46px; border-radius:10px; background:linear-gradient(135deg,#2a2a2a,#1a1a1a); display:flex; align-items:center; justify-content:center; font-weight:800; color:{GREEN}; font-size:15px; border:1px solid #2e2e2e; }}
    .fp-team-name {{ font-weight:700; font-size:14px; }}
    .fp-team-role {{ color:{GREEN}; font-size:12px; }}

    /* ==== DASHBOARD ==== */
    .fp-dash-title {{ font-size:26px; font-weight:800; margin-bottom:2px; }}
    .fp-dash-sub {{ color:{MUTED}; font-size:13.5px; margin-bottom:16px; }}
    .fp-chip {{ background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:10px; padding:9px 14px; font-size:13px; color:#e5e5e5; display:inline-flex; align-items:center; gap:6px; }}
    .fp-filterbar {{ background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:16px; padding:14px 16px; display:flex; gap:12px; margin:14px 0 18px 0; flex-wrap:wrap; }}
    .data-scope {{ display:grid; grid-template-columns:1.4fr 1fr 1fr 1fr; gap:1px; background:{CARD_BORDER}; border:1px solid {CARD_BORDER}; border-radius:14px; overflow:hidden; margin:0 0 18px; }}
    .data-scope-item {{
        background:{CARD_BG}; padding:12px 15px;
        transition:background .18s ease, filter .18s ease;
    }}
    .data-scope-item:hover {{ background:#19191c; filter:brightness(1.08); }}
    .data-scope-label {{ color:{MUTED2}; font-size:10px; text-transform:uppercase; letter-spacing:.8px; }}
    .data-scope-value {{ color:#e7ebf2; font-size:13px; font-weight:700; margin-top:5px; }}
    @media (max-width: 800px) {{ .data-scope {{ grid-template-columns:1fr 1fr; }} }}

    .kpi-card {{ background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:18px; padding:18px; }}
    .kpi-top {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:14px; }}
    .kpi-icon {{ width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:17px; }}
    .kpi-pill {{ font-size:12px; font-weight:800; padding:5px 10px; border-radius:999px; }}
    .kpi-pill.up {{ background:rgba(163,230,53,0.14); color:{GREEN}; }}
    .kpi-pill.down {{ background:rgba(248,113,113,0.14); color:{RED}; }}
    .kpi-label {{ color:{MUTED}; font-size:13px; margin-bottom:4px; }}
    .kpi-value {{ font-size:24px; font-weight:900; margin-bottom:4px; }}
    .kpi-note {{ color:{MUTED2}; font-size:11.5px; }}

    .chart-card {{ background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:18px; padding:18px 20px; height:100%; }}
    .chart-head {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:4px; }}
    .chart-title {{ font-size:15.5px; font-weight:800; margin:0; }}
    .chart-sub {{ color:{MUTED}; font-size:12.5px; margin-top:2px; }}
    .legend-dot {{ display:inline-flex; align-items:center; gap:6px; font-size:12px; color:{MUTED}; margin-left:14px; }}
    .legend-dot .sw {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
    .legend-list-item {{ display:flex; align-items:center; justify-content:space-between; padding:7px 0; font-size:13px; }}
    .legend-list-item .left {{ display:flex; align-items:center; gap:8px; color:#e5e5e5; }}
    .legend-list-item .sw {{ width:9px; height:9px; border-radius:50%; }}
    .legend-list-item .pct {{ font-weight:800; }}
    .donut-center {{ text-align:center; margin-top:-172px; margin-bottom:130px; pointer-events:none; }}
    .donut-center .num {{ font-size:22px; font-weight:900; }}
    .donut-center .lab {{ font-size:11px; color:{MUTED}; }}

    .rank-badge {{ width:26px; height:26px; border-radius:7px; background:#1a1a1c; border:1px solid {CARD_BORDER}; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:800; color:#ddd; }}
    .prod-row {{
        padding:10px 0; border-bottom:1px solid #1c1c1e;
        transition:background .18s ease, padding .18s ease;
    }}
    .prod-row:hover {{ background:rgba(255,255,255,0.035); padding-left:8px; padding-right:8px; }}
    .prod-row:last-child {{ border-bottom:none; }}
    .prod-top {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }}
    .prod-left {{ display:flex; align-items:center; gap:10px; }}
    .prod-name {{ font-weight:700; font-size:13.5px; }}
    .prod-cat {{ font-size:11.5px; color:{MUTED}; }}
    .prod-value {{ font-weight:800; font-size:13.5px; }}
    .prod-bar-bg {{ background:#1c1c1e; border-radius:999px; height:5px; width:100%; overflow:hidden; }}
    .prod-bar-fill {{ height:100%; border-radius:999px; }}
    .view-all {{ color:{GREEN}; font-size:12.5px; font-weight:700; }}
    .fp-footer {{ display:flex; justify-content:space-between; color:{MUTED2}; font-size:12px; padding:18px 4px 6px 4px; }}
    .reports-card {{ padding:0; overflow:hidden; }}
    .reports-heading {{ padding:22px 20px 18px; margin:0; }}
    .reports-table-head, .reports-row {{ display:grid; grid-template-columns:2fr 1.05fr .85fr 1.15fr .8fr 1.05fr; align-items:center; column-gap:18px; }}
    .reports-table-head {{ background:#0b0b0d; border-top:1px solid #1d1e21; border-bottom:1px solid #1d1e21; color:#8190a9; font-size:10px; letter-spacing:.7px; padding:13px 20px; }}
    .reports-row {{
        min-height:60px; padding:0 20px; border-bottom:1px solid #242529;
        color:#aab5ca; font-size:12px;
        transition:background .18s ease;
    }}
    .reports-row:hover {{ background:rgba(255,255,255,0.035); }}
    .reports-row:last-child {{ border-bottom:none; }}
    .reports-row strong {{ color:#f4f6fa; font-size:12px; }}
    .reports-region {{ display:flex; align-items:center; gap:12px; color:#dce2ee; }}
    .region-code {{ width:28px; height:28px; display:grid; place-items:center; border-radius:8px; background:rgba(163,230,53,.12); color:{GREEN}; font-size:11px; font-weight:800; }}
    .growth-pill {{ justify-self:start; border-radius:999px; padding:6px 11px; font-weight:700; font-size:11px; white-space:nowrap; }}
    .growth-positive {{ color:#08734b; background:#eafff5; }} .growth-negative {{ color:#c83d67; background:#fff1f4; }}
    @media (max-width: 800px) {{ .reports-table-head, .reports-row {{ grid-template-columns:1.8fr 1fr .8fr; }} .reports-table-head span:nth-child(n+4), .reports-row > span:nth-child(n+4) {{ display:none; }} }}

    /* ==== SIMULATION ==== */
    div[class*="st-key-scenario-wrap"] {{
        background:{CARD_BG}; border:1px solid {CARD_BORDER}; border-radius:14px; padding:4px;
    }}
    div[class*="st-key-scenA-"] button {{
        background:{GREEN} !important; color:black !important; font-weight:800 !important;
        border-radius:999px !important; padding:6px 5px !important; font-size:12px !important;
    }}
    div[class*="st-key-scenI-"] button {{
        background:transparent !important; color:{MUTED} !important; font-weight:600 !important;
        border-radius:999px !important; padding:6px 5px !important; font-size:12px !important;
    }}
    div[class*="st-key-scenA-"] button,
    div[class*="st-key-scenI-"] button {{ white-space:nowrap !important; }}
    div[class*="st-key-scenI-"] button:hover {{ color:#e5e5e5 !important; }}

    .fp-eyebrow-sm {{ display:flex; align-items:center; gap:7px; color:{GREEN}; font-weight:800; font-size:12px; letter-spacing:1.3px; margin-bottom:10px; }}
    .fp-sim-title {{ font-size:22px; font-weight:800; margin:0 0 4px 0; }}
    .fp-sim-sub {{ color:{MUTED}; font-size:13.5px; }}
    .fp-base-pill {{ background:rgba(163,230,53,0.14); color:{GREEN}; font-weight:800; font-size:12px; padding:6px 12px; border-radius:999px; }}
    .sim-stat-box {{
        background:#0d0d0d; border:1px solid {CARD_BORDER}; border-radius:14px;
        padding:12px 16px; flex:1;
        transition:transform .18s ease, background .18s ease, border-color .18s ease,
            box-shadow .18s ease, filter .18s ease;
    }}
    .sim-stat-box:hover {{
        transform:translateY(-3px); background:#151518; border-color:rgba(163,230,53,0.48);
        filter:brightness(1.08); box-shadow:0 10px 22px rgba(0,0,0,0.24);
    }}
    .sim-stat-label {{ color:{MUTED}; font-size:12.5px; margin-bottom:6px; }}
    .sim-stat-value {{ font-size:21px; font-weight:800; }}
    .sim-stat-value.pos {{ color:{GREEN}; }}

    .outlook-head-note {{ color:{MUTED2}; font-size:10.5px; letter-spacing:1px; font-weight:700; text-transform:uppercase; }}
    .outlook-row {{
        padding:11px 0; border-bottom:1px solid #1c1c1e;
        transition:background .18s ease, padding .18s ease;
    }}
    .outlook-row:hover {{ background:rgba(255,255,255,0.035); padding-left:8px; padding-right:8px; }}
    .outlook-row:last-child {{ border-bottom:none; }}
    .outlook-top {{ display:flex; align-items:flex-start; justify-content:space-between; margin-bottom:8px; }}
    .outlook-name {{ font-weight:700; font-size:13.5px; }}
    .outlook-status {{ font-size:11.5px; font-weight:600; margin-top:2px; }}
    .outlook-value {{ font-weight:800; font-size:13.5px; text-align:right; }}
    .outlook-forecast {{ font-size:11.5px; font-weight:700; text-align:right; margin-top:2px; }}
    .outlook-forecast.pos {{ color:{GREEN}; }}
    .outlook-forecast.neg {{ color:{RED}; }}

    .insight-box {{
        background:rgba(163,230,53,0.08); border:1px solid rgba(163,230,53,0.35);
        border-radius:14px; padding:14px 16px; margin-top:14px; font-size:13px; color:#dcdcdc;
        display:flex; align-items:flex-start; gap:10px; line-height:1.5;
    }}
    .insight-box b {{ color:{GREEN}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

if st.session_state.dashboard_light_mode:
    st.markdown(
        """
        <style>
        .stApp { background:#f2f5ef !important; color:#18201b !important; }
        section[data-testid="stSidebar"] { background:#e8ede5 !important; border-color:#d2dbcf !important; }
        .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMarkdownContainer"] p,
        .fp-dash-title, .fp-sim-title, .chart-title, .kpi-value, .kpi-label,
        .fp-card-value, .sim-stat-value, .prod-name, .prod-value, .outlook-name,
        .outlook-value, .reports-row strong, .fp-logo-text b { color:#18201b !important; }
        .fp-dash-sub, .chart-sub, .sim-stat-label, .kpi-note, .prod-cat,
        .outlook-head-note, .fp-logo-text span { color:#59665c !important; }
        .fp-card, .kpi-card, .chart-card, .sim-stat-box, .fp-chip, .fp-filterbar,
        .fp-search, .fp-iconbtn, .fp-stat-box, .fp-team-card, .data-scope-item {
            background:#ffffff !important; border-color:#d5ded3 !important; color:#18201b !important;
        }
        .reports-table-head { background:#e8ede5 !important; color:#59665c !important; }
        .reports-row, .prod-row, .outlook-row { color:#39443b !important; border-color:#e1e7df !important; }
        div[data-testid="stButton"] button,
        div[data-testid="stPopover"] > button,
        div[data-testid="stPopover"] button {
            background:#ffffff !important; color:#263229 !important; border-color:#d5ded3 !important;
        }
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
        div[data-testid="stTextInput"] input,
        div[data-testid="stPopoverBody"], div[data-baseweb="popover"],
        div[data-baseweb="menu"], ul[role="listbox"] {
            background:#ffffff !important; color:#263229 !important; border-color:#d5ded3 !important;
        }
        li[role="option"] { color:#263229 !important; }
        li[role="option"]:hover { background:#e8ede5 !important; }
        div[class*="st-key-cta-btn"] button,
        div[class*="st-key-export-btn"] button,
        div[class*="st-key-navA-settings"] button,
        div[class*="st-key-scenA-"] button {
            background:#a3e635 !important; color:#101510 !important; border-color:#a3e635 !important;
        }
        div[class*="st-key-navI-"] button { color:#39443b !important; }
        div[class*="st-key-navI-"] button:hover { background:#dbe5d7 !important; color:#18201b !important; }
        div[class*="st-key-dashboard-home"] button,
        div[class*="st-key-sidebar-toggle"] button,
        div[class*="st-key-theme-toggle"] button {
            color:#263229 !important; background:#ffffff !important; border-color:#d5ded3 !important;
        }
        div[class*="st-key-dashboard-sidebar-panel"] {
            background:rgba(232,237,229,0.9) !important; border-color:#d5ded3 !important;
        }
        div[class*="st-key-dashboard-sticky-header"] {
            background:rgba(242,245,239,0.82) !important;
            border-color:rgba(89,102,92,0.2) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# CHART HELPERS
# ----------------------------------------------------------------------
def landing_sales_chart(values, months, highlight_from):
    data = pd.DataFrame({
        "index": np.arange(len(values)),
        "month": months,
        "sales": np.asarray(values, dtype=float),
        "period": ["Earlier" if index < highlight_from else "Recent" for index in range(len(values))],
    })
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X("index:Q", axis=None, scale=alt.Scale(domain=[-0.5, len(values) - 0.5])),
        y=alt.Y("sales:Q", axis=None, scale=alt.Scale(zero=True)),
        color=alt.Color("period:N", scale=alt.Scale(
            domain=["Earlier", "Recent"], range=["#3f3f46", GREEN],
        ), legend=None),
        tooltip=[
            alt.Tooltip("month:N", title="Month"),
            alt.Tooltip("period:N", title="Period"),
            alt.Tooltip("sales:Q", title="Sales ($K)", format=",.1f"),
        ],
    )
    return chart.properties(height=165).configure(background="transparent").configure_view(stroke=None)


def revenue_bar_chart(months, revenue, expenses):
    data = pd.DataFrame([
        {"month": month, "series": series, "value": value}
        for month, revenue_value, expense_value in zip(months, revenue, expenses)
        for series, value in (("Revenue", revenue_value), ("Expenses", expense_value))
    ])
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X("month:N", sort=list(months), axis=alt.Axis(
            labelColor=MUTED, domain=False, ticks=False, title=None,
        )),
        xOffset=alt.XOffset("series:N"),
        y=alt.Y("value:Q", axis=alt.Axis(
            labelColor=MUTED, domain=False, ticks=False,
            labelExpr="'$' + format(datum.value, ',.0f') + 'K'", title=None,
        ), scale=alt.Scale(zero=True)),
        color=alt.Color("series:N", scale=alt.Scale(
            domain=["Revenue", "Expenses"], range=[GREEN, LAVENDER],
        ), legend=None),
        tooltip=[
            alt.Tooltip("month:N", title="Month"),
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("value:Q", title="Amount ($K)", format=",.1f"),
        ],
    )
    return chart.properties(height=225).configure(background="transparent").configure_view(stroke=None)


def traffic_donut(labels, values, colors, total):
    data = pd.DataFrame({
        "source": labels,
        "share": values,
    })
    arcs = alt.Chart(data).mark_arc(
        innerRadius=58, outerRadius=96, stroke=PAGE_BG, strokeWidth=2,
    ).encode(
        theta=alt.Theta("share:Q", stack=True),
        color=alt.Color("source:N", scale=alt.Scale(domain=labels, range=colors), legend=None),
        tooltip=[
            alt.Tooltip("source:N", title="Source"),
            alt.Tooltip("share:Q", title="Share", format=".0f", formatType="number"),
        ],
    )
    center = alt.Chart(pd.DataFrame({"total": [total]})).mark_text(
        color="white", fontSize=20, fontWeight="bold",
    ).encode(text="total:N")
    return (arcs + center).properties(height=240).configure(
        background="transparent",
    ).configure_view(stroke=None)


def simulation_chart(hist, forecast, band_max=9, hist_months=None, forecast_months=None):
    """Interactive historical and forecast chart with hoverable values and ranges."""
    n_hist = len(hist)
    forecast_line = np.concatenate(([hist[-1]], forecast)) / 1000
    periods = list(hist_months or ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"][-n_hist:])
    periods.extend(forecast_months or ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb"][:len(forecast)])
    split_index = n_hist - 1
    forecast_indexes = np.arange(split_index, n_hist + len(forecast))
    band = np.linspace(0, band_max / 1000, len(forecast_line))
    lower = np.maximum(forecast_line - band, 0)
    upper = forecast_line + band
    chart_top = float(max(max(hist) / 1000, max(upper)) * 1.2)
    tick_indexes = list(range(0, len(periods), 2))
    label_expression = f"{json.dumps(periods)}[datum.value]"
    x_axis = alt.X(
        "index:Q",
        scale=alt.Scale(domain=[0, len(periods) - 1]),
        axis=alt.Axis(values=tick_indexes, labelExpr=label_expression, labelColor=BLUE,
                      domain=False, ticks=False, grid=False, title=None),
    )
    y_axis = alt.Axis(labels=False, ticks=False, domain=False, grid=True,
                      gridColor="#2a2a2c", title=None)
    y_scale = alt.Scale(domain=[0, chart_top])

    forecast_data = pd.DataFrame({
        "index": forecast_indexes,
        "period": periods[split_index:],
        "value": forecast_line,
        "lower": lower,
        "upper": upper,
        "series": ["Last actual", *(["Forecast"] * len(forecast))],
    })
    history_data = pd.DataFrame({
        "index": np.arange(n_hist),
        "period": periods[:n_hist],
        "value": np.asarray(hist, dtype=float) / 1000,
    })
    band_chart = alt.Chart(forecast_data).mark_area(color="#332a5e", opacity=0.5).encode(
        x=x_axis,
        y=alt.Y("lower:Q", scale=y_scale, axis=y_axis),
        y2="upper:Q",
        tooltip=[
            alt.Tooltip("period:N", title="Month"),
            alt.Tooltip("lower:Q", title="Forecast low ($M)", format=",.2f"),
            alt.Tooltip("upper:Q", title="Forecast high ($M)", format=",.2f"),
        ],
    )
    history_chart = alt.Chart(history_data).mark_line(
        color=GREEN, strokeWidth=2.5,
        point=alt.OverlayMarkDef(filled=True, fill="white", stroke=GREEN, size=42),
    ).encode(
        x=x_axis,
        y=alt.Y("value:Q", scale=y_scale, axis=y_axis),
        tooltip=[
            alt.Tooltip("period:N", title="Month"),
            alt.Tooltip("value:Q", title="Actual sales ($M)", format=",.2f"),
        ],
    )
    forecast_chart = alt.Chart(forecast_data).mark_line(
        color=TEAL, strokeWidth=2.3, strokeDash=[5, 4],
        point=alt.OverlayMarkDef(filled=True, fill=TEAL, size=38),
    ).encode(
        x=x_axis,
        y=alt.Y("value:Q", scale=y_scale, axis=y_axis),
        tooltip=[
            alt.Tooltip("period:N", title="Month"),
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("value:Q", title="Sales ($M)", format=",.2f"),
            alt.Tooltip("lower:Q", title="Forecast low ($M)", format=",.2f"),
            alt.Tooltip("upper:Q", title="Forecast high ($M)", format=",.2f"),
        ],
    )
    split_rule = alt.Chart(pd.DataFrame({"index": [split_index + 0.5]})).mark_rule(
        color="#4b4b52", strokeDash=[3, 3],
    ).encode(x="index:Q")
    labels = pd.DataFrame({
        "index": [split_index / 2, split_index + 0.5 + len(forecast) / 2],
        "level": [chart_top * 0.96, chart_top * 0.96],
        "section": ["HISTORICAL", "FORECAST"],
    })
    section_labels = alt.Chart(labels).mark_text(fontSize=11, fontWeight="bold").encode(
        x="index:Q", y="level:Q", text="section:N",
        color=alt.Color("section:N", scale=alt.Scale(
            domain=["HISTORICAL", "FORECAST"], range=[BLUE, GREEN],
        ), legend=None),
    )
    return alt.layer(band_chart, split_rule, history_chart, forecast_chart, section_labels).properties(
        height=230,
    ).configure(background="transparent").configure_view(stroke=None)


def business_perf_chart(months, revenue, profit):
    indexes = np.arange(len(months))
    revenue_data = pd.DataFrame({
        "index": indexes, "month": months,
        "value": np.asarray(revenue, dtype=float) / 1000,
    })
    profit_data = pd.DataFrame({
        "index": indexes, "month": months,
        "value": np.asarray(profit, dtype=float) / 1000,
    })
    chart_top = max(float(revenue_data["value"].max()), float(profit_data["value"].max())) * 1.25
    ticks = list(range(0, len(months), 2))
    x_axis = alt.X(
        "index:Q", scale=alt.Scale(domain=[0, len(months) - 1]),
        axis=alt.Axis(values=ticks, labelExpr=f"{json.dumps(list(months))}[datum.value]",
                      labelColor=MUTED, domain=False, ticks=False, grid=False, title=None),
    )
    y_scale = alt.Scale(domain=[0, chart_top])
    y_axis = alt.Axis(labels=False, ticks=False, domain=False, grid=True,
                      gridColor="#2a2a2c", title=None)
    revenue_area = alt.Chart(revenue_data).mark_area(color="#2a2350", opacity=0.55).encode(
        x=x_axis,
        y=alt.Y("value:Q", scale=y_scale, axis=y_axis),
        y2=alt.value(0),
    )
    revenue_line = alt.Chart(revenue_data).mark_line(
        color=GREEN, strokeWidth=2.4,
        point=alt.OverlayMarkDef(filled=True, fill="white", stroke=GREEN, size=42),
    ).encode(
        x=x_axis,
        y=alt.Y("value:Q", scale=y_scale, axis=y_axis),
        tooltip=[
            alt.Tooltip("month:N", title="Month"),
            alt.Tooltip("value:Q", title="Revenue ($M)", format=",.2f"),
        ],
    )
    profit_line = alt.Chart(profit_data).mark_line(
        color=TEAL, strokeWidth=2, strokeDash=[5, 4],
        point=alt.OverlayMarkDef(filled=True, fill=TEAL, size=38),
    ).encode(
        x=x_axis,
        y=alt.Y("value:Q", scale=y_scale, axis=y_axis),
        tooltip=[
            alt.Tooltip("month:N", title="Month"),
            alt.Tooltip("value:Q", title="Net profit ($M)", format=",.2f"),
        ],
    )
    return alt.layer(revenue_area, revenue_line, profit_line).properties(
        height=220,
    ).configure(background="transparent").configure_view(stroke=None)


# ----------------------------------------------------------------------
# DATA LOADING / COMPUTED STATS FROM AN UPLOADED DATASET
# ----------------------------------------------------------------------
def load_dataframe(file):
    if isinstance(file, (str, Path)):
        path = Path(file).resolve()
        try:
            modified_ns = path.stat().st_mtime_ns
        except OSError as error:
            st.error(f"Could not access file: {error}")
            return None
        df, error_message = _load_dataframe_from_path(str(path), modified_ns)
    elif isinstance(file, tuple) and len(file) == 2:
        df, error_message = _load_dataframe_from_bytes(file[0], file[1])
    else:
        df, error_message = _load_dataframe_from_bytes(file.name, file.getvalue())

    if error_message:
        st.error(f"Could not read file: {error_message}")
        return None
    return df


@st.cache_data(show_spinner=False, max_entries=4)
def _load_dataframe_from_path(path_string: str, modified_ns: int):
    path = Path(path_string)
    df, error_message = _parse_dataframe_bytes(path.name, path.read_bytes())
    if df is not None:
        df.attrs["source_modified_ns"] = modified_ns
    return df, error_message


@st.cache_data(show_spinner=False, max_entries=4)
def _load_dataframe_from_bytes(file_name: str, file_bytes: bytes):
    return _parse_dataframe_bytes(file_name, file_bytes)


def _parse_dataframe_bytes(file_name: str, file_bytes: bytes):
    try:
        if file_name.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(file_bytes))
        else:
            preview = pd.read_excel(BytesIO(file_bytes), header=None, nrows=30)
            header_row = next(
                (
                    index
                    for index, row in preview.iterrows()
                    if {str(value).strip().lower() for value in row.dropna()}
                    >= {"product", "invoice date", "total sales"}
                ),
                0,
            )
            df = pd.read_excel(BytesIO(file_bytes), header=header_row)
    except Exception as error:
        return None, str(error)
    df = df.loc[:, ~df.columns.astype(str).str.lower().str.startswith("unnamed")]
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {
        "invoice date": "date",
        "total sales": "sales",
        "operating profit": "profit",
        "units sold": "units_sold",
        "sales method": "channel",
        "region": "region",
    }
    df = df.rename(columns=aliases)
    return df, None


def filter_dashboard_dataframe(df: pd.DataFrame, date_range: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    return _filter_dashboard_dataframe_cached(
        df,
        date_range,
        st.session_state.get("dashboard_region", "All regions"),
        st.session_state.get("dashboard_channel", "All channels"),
        st.session_state.get("dashboard_product", "All products"),
        st.session_state.get("dashboard_retailer", "All retailers"),
        st.session_state.get("dashboard_state", "All states"),
        st.session_state.get("dashboard_city", "All cities"),
    )


@st.cache_data(show_spinner=False, max_entries=32)
def _filter_dashboard_dataframe_cached(df: pd.DataFrame, date_range: str,
                                       selected_region: str, selected_channel: str,
                                       selected_product: str, selected_retailer: str,
                                       selected_state: str, selected_city: str) -> pd.DataFrame:
    filtered = df.copy()
    if "date" in filtered.columns:
        filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce")
        latest_date = filtered["date"].max()
        if pd.notna(latest_date) and date_range != "All time":
            if date_range == "Last 30 days":
                start_date = latest_date - pd.Timedelta(days=29)
            elif date_range == "Last 90 days":
                start_date = latest_date - pd.Timedelta(days=89)
            elif date_range == "YTD":
                start_date = pd.Timestamp(year=latest_date.year, month=1, day=1)
            else:
                start_date = latest_date.to_period("M").start_time - pd.DateOffset(months=11)
            filtered = filtered[filtered["date"].between(start_date, latest_date)]

    region_column = "market" if "market" in filtered.columns else "region" if "region" in filtered.columns else None
    if region_column:
        if selected_region != "All regions":
            filtered = filtered[filtered[region_column].astype(str) == selected_region]

    if "channel" in filtered.columns:
        if selected_channel != "All channels":
            filtered = filtered[filtered["channel"].astype(str) == selected_channel]

    for column, selected_value, default_value in (
        ("product", selected_product, "All products"),
        ("retailer", selected_retailer, "All retailers"),
        ("state", selected_state, "All states"),
        ("city", selected_city, "All cities"),
    ):
        if column in filtered.columns and selected_value != default_value:
            filtered = filtered[filtered[column].astype(str) == selected_value]

    return filtered


@st.cache_data(show_spinner=False)
def train_sales_forecast(monthly_sales: tuple[float, ...], horizon: int = 7) -> dict:
    values = np.asarray(monthly_sales, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("At least two monthly sales values are needed to train a forecast.")

    seasonal = len(values) >= 24
    has_trend = len(values) >= 3
    model = ExponentialSmoothing(
        values,
        trend="add" if has_trend else None,
        damped_trend=has_trend,
        seasonal="add" if seasonal else None,
        seasonal_periods=12 if seasonal else None,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True)
    forecast = np.maximum(np.asarray(fitted.forecast(horizon), dtype=float), 0)
    residuals = np.asarray(fitted.resid, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    rmse = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else 0.0
    scale = max(float(np.mean(np.abs(values))), 1.0)

    return {
        "forecast_series": np.round(forecast, 1).tolist(),
        "band_max": round(max(rmse * np.sqrt(horizon), 1.0), 1),
        "model_fit": int(round(max(0.0, min(100.0, (1 - rmse / scale) * 100)))),
        "forecast_model": "Holt-Winters seasonal" if seasonal else "Damped Holt trend",
    }


def half_period_change(values):
    period_values = np.asarray(values, dtype=float)
    period_values = period_values[np.isfinite(period_values)]
    if len(period_values) < 2:
        return None

    midpoint = max(1, len(period_values) // 2)
    first_period_mean = float(np.mean(period_values[:midpoint]))
    second_period_mean = float(np.mean(period_values[midpoint:]))
    if first_period_mean == 0:
        return None
    return round((second_period_mean - first_period_mean) / abs(first_period_mean) * 100, 1)


def forecast_sales_by_group(df: pd.DataFrame, group_column: str, horizon: int = 7) -> dict:
    required_columns = {"date", "sales", group_column}
    if not required_columns.issubset(df.columns):
        return {}

    forecast_frame = df[["date", "sales", group_column]].copy()
    forecast_frame["date"] = pd.to_datetime(forecast_frame["date"], errors="coerce")
    forecast_frame["sales"] = pd.to_numeric(forecast_frame["sales"], errors="coerce")
    forecast_frame = forecast_frame.dropna(subset=["date", "sales", group_column])
    forecasts = {}

    for group_name, group in forecast_frame.groupby(group_column):
        monthly_sales = group.set_index("date").resample("MS")["sales"].sum()
        if len(monthly_sales) < 2:
            continue

        monthly_values = (monthly_sales.to_numpy(dtype=float) / 1000).tolist()
        model_output = train_sales_forecast(tuple(monthly_values), horizon=horizon)
        trailing_count = min(horizon, len(monthly_values))
        baseline = float(np.mean(monthly_values[-trailing_count:]))
        forecast_mean = float(np.mean(model_output["forecast_series"]))
        growth = round((forecast_mean / baseline - 1) * 100, 1) if baseline > 0 else None
        forecasts[str(group_name)] = {
            **model_output,
            "baseline_sales": round(baseline, 1),
            "last_actual": round(monthly_values[-1], 1),
            "projected_sales": round(sum(model_output["forecast_series"]) / 1000, 2),
            "forecast": growth,
        }

    return forecasts


@st.cache_data(show_spinner=False, max_entries=32)
def compute_stats(df: pd.DataFrame) -> dict:
    """Derive every dashboard figure from an uploaded dataset, falling
    back to the static demo value for anything the dataset can't supply."""
    s = {k: (v.copy() if isinstance(v, (list, dict)) else v) for k, v in DEFAULTS.items()}
    s.update({
        "kpi_revenue_delta": None,
        "kpi_profit_delta": None,
        "kpi_orders_delta": None,
        "kpi_conv": None,
        "kpi_conv_delta": None,
        "kpi_conv_label": "Conversion unavailable",
        "kpi_orders_label": "Sales records",
        "kpi_delta_note": None,
        "kpi_conv_delta_suffix": "%",
    })

    cols = set(df.columns)
    s["data_rows"] = int(len(df))
    has_sales = "sales" in cols
    has_date = "date" in cols
    has_product = "product" in cols
    has_market = "market" in cols or "region" in cols
    region_col = "market" if "market" in cols else "region" if "region" in cols else None
    has_profit = "profit" in cols
    has_units = "units_sold" in cols
    has_expenses = "expenses" in cols or "cost" in cols
    has_channel = next((column for column in ["channel", "source", "traffic_source"] if column in cols), None)
    has_orders = next((column for column in ["order_id", "orders", "order"] if column in cols), None)
    has_visitors = next((column for column in ["visitors", "sessions", "visits"] if column in cols), None)

    if has_product:
        s["products"] = int(df["product"].nunique())
    if has_market:
        s["markets"] = int(df[region_col].nunique())
        s["region_count"] = s["markets"]

    if has_date:
        parsed_dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if not parsed_dates.empty:
            s["date_range"] = f"{parsed_dates.min():%b %Y} – {parsed_dates.max():%b %Y}"

    if has_sales and has_date:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d["sales"] = pd.to_numeric(d["sales"], errors="coerce")
        if has_profit:
            d["profit"] = pd.to_numeric(d["profit"], errors="coerce")
        d = d.dropna(subset=["date", "sales"])

        if len(d) >= 2:
            monthly = d.set_index("date").resample("MS")["sales"].sum()
            monthly = monthly[monthly.index.notna()]

            if len(monthly) >= 2:
                x = np.arange(len(monthly))
                y = monthly.values.astype(float)
                coeffs = np.polyfit(x, y, 1)
                trend = np.polyval(coeffs, x)

                recent = y[-3:] if len(y) >= 3 else y
                projected_annual = float(np.mean(recent)) * 12
                s["projected_sales"] = round(projected_annual / 1_000_000, 2)
                s["kpi_revenue"] = round(float(np.sum(y)) / 1_000_000, 2)

                half = max(1, len(y) // 2)
                fh, sh = np.mean(y[:half]), (np.mean(y[half:]) if len(y) > half else np.mean(y))
                if fh > 0:
                    growth = (sh - fh) / fh * 100
                    s["growth_pct"] = round(float(growth), 1)
                    s["kpi_revenue_delta"] = round(float(growth), 1)

                ss_res = float(np.sum((y - trend) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = max(0.0, min(1.0, 1 - ss_res / ss_tot if ss_tot > 0 else 0))
                s["confidence"] = int(round(r2 * 100))

                bar_vals = (monthly.values[-11:] / 1000).round(1).tolist()
                s["bar_values"] = bar_vals
                s["bar_months"] = monthly.index[-len(bar_vals):].strftime("%b %Y").tolist()
                s["bar_highlight_from"] = max(0, len(bar_vals) - 4)

                # Revenue overview / business performance chart data (last 12 months)
                last12 = monthly[-12:]
                s["months"] = [dt.strftime("%b") for dt in last12.index]
                rev_k = (last12.values / 1000).round(1)
                s["revenue_series"] = rev_k.tolist()
                s["perf_revenue"] = rev_k.tolist()

                if has_profit:
                    profit_monthly = d.set_index("date").resample("MS")["profit"].sum()
                    profit_last12 = profit_monthly.reindex(last12.index, fill_value=0)
                    profit_k = (profit_last12.values / 1000).round(1)
                    s["perf_profit"] = profit_k.tolist()
                    s["expenses_series"] = np.maximum(rev_k - profit_k, 0).round(1).tolist()
                    s["kpi_profit"] = round(float(d["profit"].sum()) / 1_000_000, 2)
                    s["kpi_profit_suffix"] = "M"
                    s["kpi_profit_label"] = "Net profit"
                    s["expenses_label"] = "Expenses"
                    profit_all_months = profit_monthly.reindex(monthly.index, fill_value=0).to_numpy(dtype=float)
                    s["kpi_profit_delta"] = half_period_change(profit_all_months)
                    s["kpi_conv"] = round(float(d["profit"].sum() / max(d["sales"].sum(), 1) * 100), 1)
                    s["kpi_conv_label"] = "Operating margin"
                    midpoint = max(1, len(y) // 2)
                    first_margin = float(profit_all_months[:midpoint].sum() / max(y[:midpoint].sum(), 1) * 100)
                    second_margin = float(profit_all_months[midpoint:].sum() / max(y[midpoint:].sum(), 1) * 100)
                    s["kpi_conv_delta"] = round(second_margin - first_margin, 1)
                    s["kpi_conv_delta_suffix"] = " pp"
                    s["kpi_delta_note"] = "second half vs first half"
                elif has_expenses:
                    s["kpi_profit_label"] = "Net profit"
                    s["expenses_label"] = "Expenses"
                    exp_col = "expenses" if "expenses" in cols else "cost"
                    d2 = df.copy()
                    d2["date"] = pd.to_datetime(d2["date"], errors="coerce")
                    d2[exp_col] = pd.to_numeric(d2[exp_col], errors="coerce")
                    d2 = d2.dropna(subset=["date", exp_col])
                    exp_monthly = d2.set_index("date").resample("MS")[exp_col].sum()[-12:]
                    exp_k = (exp_monthly.values / 1000).round(1)
                    if len(exp_k) == len(rev_k):
                        s["expenses_series"] = exp_k.tolist()
                else:
                    s["kpi_profit_label"] = "Estimated net profit"
                    s["expenses_label"] = "Estimated expenses"
                    s["expenses_series"] = (rev_k * 0.73).round(1).tolist()

                profit_k = (rev_k - np.array(s["expenses_series"])).clip(min=0)
                if not has_profit:
                    s["perf_profit"] = profit_k.round(1).tolist()
                    margin = 1 - (np.sum(s["expenses_series"]) / max(np.sum(rev_k), 1))
                    s["kpi_profit"] = round(float(np.sum(rev_k) * max(margin, 0.05)), 1)
                    s["kpi_profit_delta"] = half_period_change(profit_k) if has_expenses else None
                    if not has_expenses:
                        s["kpi_delta_note"] = "Profit and expenses estimated"

                hist = (monthly.values[-7:] / 1000).round(1).tolist()
                model_output = train_sales_forecast(tuple((monthly.values / 1000).astype(float)))
                baseline_sales = float(np.mean(monthly.values[-min(7, len(monthly)):] / 1000))
                forecast_mean = float(np.mean(model_output["forecast_series"]))
                forecast_growth = (
                    round((forecast_mean / baseline_sales - 1) * 100, 1)
                    if baseline_sales > 0 else None
                )
                history_months = monthly.index[-7:].strftime("%b %Y").tolist()
                future_index = pd.date_range(
                    monthly.index[-1] + pd.offsets.MonthBegin(1), periods=7, freq="MS"
                )
                s.update({
                    "hist_series": hist,
                    "hist_months": history_months,
                    "forecast_months": future_index.strftime("%b %Y").tolist(),
                    "forecast_growth_pct": forecast_growth,
                    **model_output,
                })
            elif len(monthly) == 1:
                last_month = monthly.index[-1]
                revenue_total = float(monthly.iloc[-1])
                revenue_k = round(revenue_total / 1000, 1)
                s["kpi_revenue"] = round(revenue_total / 1_000_000, 2)
                s["projected_sales"] = round(revenue_total * 12 / 1_000_000, 2)
                s["growth_pct"] = 0.0
                s["kpi_revenue_delta"] = 0.0
                s["months"] = [last_month.strftime("%b")]
                s["revenue_series"] = [revenue_k]
                s["perf_revenue"] = [revenue_k]
                s["bar_values"] = [revenue_k]
                s["bar_months"] = [last_month.strftime("%b %Y")]
                s["bar_highlight_from"] = 0

                if has_profit:
                    profit_total = float(d["profit"].sum())
                    profit_k = round(profit_total / 1000, 1)
                    s["kpi_profit"] = round(profit_total / 1_000_000, 2)
                    s["kpi_profit_suffix"] = "M"
                    s["kpi_profit_label"] = "Net profit"
                    s["perf_profit"] = [profit_k]
                    s["expenses_series"] = [max(round(revenue_k - profit_k, 1), 0)]
                    s["kpi_conv"] = round(profit_total / max(revenue_total, 1) * 100, 1)
                    s["kpi_conv_label"] = "Operating margin"
                elif has_expenses:
                    s["kpi_profit_label"] = "Net profit"
                    expense_column = "expenses" if "expenses" in cols else "cost"
                    expense_total = float(pd.to_numeric(df[expense_column], errors="coerce").fillna(0).sum())
                    expense_k = round(expense_total / 1000, 1)
                    s["expenses_series"] = [expense_k]
                    profit_k = max(round(revenue_k - expense_k, 1), 0)
                    s["perf_profit"] = [profit_k]
                    s["kpi_profit"] = round(profit_k / 1000, 2)
                    s["kpi_profit_suffix"] = "M"
                else:
                    s["kpi_profit_label"] = "Estimated net profit"
                    expense_k = round(revenue_k * 0.73, 1)
                    s["expenses_series"] = [expense_k]
                    profit_k = max(round(revenue_k - expense_k, 1), 0)
                    s["perf_profit"] = [profit_k]
                    s["kpi_profit"] = round(profit_k / 1000, 2)
                    s["kpi_profit_suffix"] = "M"

    if has_orders:
        if has_orders in {"order_id", "order"} or df[has_orders].dtype == object:
            s["kpi_orders"] = int(df[has_orders].nunique())
        else:
            s["kpi_orders"] = int(pd.to_numeric(df[has_orders], errors="coerce").fillna(0).sum())
        s["kpi_orders_label"] = "Total orders"
        s["kpi_orders_delta"] = None
    elif has_units:
        s["kpi_orders"] = int(pd.to_numeric(df["units_sold"], errors="coerce").fillna(0).sum())
        s["kpi_orders_label"] = "Units sold"
        if has_date:
            order_frame = df.copy()
            order_frame["date"] = pd.to_datetime(order_frame["date"], errors="coerce")
            order_frame["units_sold"] = pd.to_numeric(order_frame["units_sold"], errors="coerce")
            order_frame = order_frame.dropna(subset=["date", "units_sold"])
            midpoint = order_frame["date"].min() + (order_frame["date"].max() - order_frame["date"].min()) / 2
            first_units = order_frame.loc[order_frame["date"] < midpoint, "units_sold"].sum()
            second_units = order_frame.loc[order_frame["date"] >= midpoint, "units_sold"].sum()
            if first_units > 0:
                s["kpi_orders_delta"] = round(float((second_units - first_units) / first_units * 100), 1)
                s["kpi_delta_note"] = "second half vs first half"
    elif has_sales:
        s["kpi_orders"] = int(len(df))
        s["kpi_orders_label"] = "Sales records"
        s["kpi_orders_delta"] = None

    if has_visitors and has_orders:
        visits = pd.to_numeric(df[has_visitors], errors="coerce").sum()
        orders = s["kpi_orders"]
        if visits and visits > 0:
            s["kpi_conv"] = round(orders / visits * 100, 2)
            s["kpi_conv_label"] = "Conversion rate"
            s["kpi_conv_delta_suffix"] = "%"

    if has_channel:
        counts = df[has_channel].astype(str).str.strip().str.title().value_counts()
        total = counts.sum()
        if total > 0:
            top = counts.head(4)
            pct = (top / total * 100).round(0).astype(int)
            s["traffic_labels"] = top.index.tolist()
            s["traffic_values"] = pct.tolist()
            palette = [GREEN, BLUE, TEAL, ORANGE, "#a78bfa", "#f472b6"]
            s["traffic_colors"] = palette[: len(top)]
            s["traffic_total"] = f"{int(total):,}"

    if has_product and has_sales:
        product_sales = df.assign(sales=pd.to_numeric(df["sales"], errors="coerce")).groupby("product")["sales"].sum().sort_values(ascending=False)
        palette = [GREEN, BLUE, TEAL, ORANGE, "#a78bfa", "#f472b6"]
        s["top_products"] = [
            {"name": str(name), "category": "Product category", "value": round(val / 1000, 1), "color": palette[i]}
            for i, (name, val) in enumerate(product_sales.head(4).items())
        ]
        s["low_products"] = [
            {"name": str(name), "category": "Product category", "value": round(val / 1000, 1), "color": palette[i % len(palette)]}
            for i, (name, val) in enumerate(product_sales.tail(2).sort_values().items())
        ]
        product_outlook = []
        total_product_sales = max(float(product_sales.sum()), 1)
        product_forecasts = forecast_sales_by_group(df, "product") if has_date else {}
        for index, (name, value) in enumerate(product_sales.items()):
            forecast = product_forecasts.get(str(name))
            product_growth = forecast["forecast"] if forecast else None
            status = (
                "Insufficient history" if product_growth is None
                else "Strong momentum" if product_growth >= 10
                else "Above trend" if product_growth >= 0
                else "At risk"
            )
            product_outlook.append({
                "name": str(name), "value": round(float(value) / 1000, 1), "status": status,
                "actual_sales": round(float(value) / 1_000_000, 2),
                "forecast": product_growth,
                "forecast_series": forecast["forecast_series"] if forecast else None,
                "baseline_sales": forecast["baseline_sales"] if forecast else None,
                "last_actual": forecast["last_actual"] if forecast else None,
                "projected_sales": forecast["projected_sales"] if forecast else None,
                "model_fit": forecast["model_fit"] if forecast else None,
                "color": palette[index % len(palette)],
                "bar_pct": max(6, round(float(value) / float(product_sales.iloc[0]) * 100)),
            })
        s["product_outlook"] = product_outlook
        s["insight_product"] = str(product_sales.index[0])
        s["insight_pct"] = round(float(product_sales.iloc[0]) / total_product_sales * 100)

    if has_market and has_sales:
        regional = df.assign(sales=pd.to_numeric(df["sales"], errors="coerce"))
        regional["date"] = pd.to_datetime(regional["date"], errors="coerce") if has_date else pd.NaT
        regional_forecasts = forecast_sales_by_group(df, region_col) if has_date else {}
        rows = []
        if has_orders:
            volume_label = "Orders"
            average_label = "Average order value"
        elif has_units:
            volume_label = "Units sold"
            average_label = "Sales per unit"
        else:
            volume_label = "Sales records"
            average_label = "Sales per record"
        s["regional_volume_label"] = volume_label
        s["regional_average_label"] = average_label
        for index, (region, group) in enumerate(regional.groupby(region_col)):
            sales_total = float(group["sales"].sum())
            if has_orders:
                if has_orders in {"order_id", "order"} or group[has_orders].dtype == object:
                    volume_total = float(group[has_orders].nunique())
                else:
                    volume_total = float(pd.to_numeric(group[has_orders], errors="coerce").sum())
            elif has_units:
                volume_total = float(pd.to_numeric(group["units_sold"], errors="coerce").sum())
            else:
                volume_total = float(len(group))
            growth = None
            if has_date:
                yearly = group.dropna(subset=["date"]).groupby(group["date"].dt.year)["sales"].sum()
                if len(yearly) >= 2 and yearly.iloc[-2] > 0:
                    growth = (yearly.iloc[-1] - yearly.iloc[-2]) / yearly.iloc[-2] * 100
            forecast = regional_forecasts.get(str(region))
            rows.append({"code": str(region)[:2].upper(), "name": str(region), "sales": sales_total,
                         "volume": int(volume_total), "average": sales_total / max(volume_total, 1),
                         "growth": round(float(growth), 1) if growth is not None else None,
                         "positive": growth >= 0 if growth is not None else None,
                         "forecast_sales": forecast["projected_sales"] if forecast else None,
                         "forecast_model": forecast["forecast_model"] if forecast else None})
        s["regional_performance"] = sorted(rows, key=lambda row: row["sales"], reverse=True)

    return s


# ----------------------------------------------------------------------
# TOP BAR (dashboard pages)
# ----------------------------------------------------------------------
def render_topbar(raw_df, filtered_df, stats, source_label):
    topbar_columns = (
        [.45, 1.6, 2.4, 3.35]
        if st.session_state.sidebar_visible
        else [.45, 1.9, 2.8, 3.1]
    )
    home_col, logo_col, search_col, status_col = st.columns(
        topbar_columns, gap="small", vertical_alignment="center"
    )
    with home_col:
        st.button("←", key="dashboard-home", on_click=go_landing, width="content")
    with logo_col:
        st.markdown(
            f"""
            <div class="fp-logo" style="padding:4px 0;">
                <div class="fp-logo-sq">✦</div>
                <div class="fp-logo-text"><b>Adidas Foot print</b><span>BUSINESS INTELLIGENCE</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with search_col:
        with st.popover("🔎 Search", help="Search rows in the active dataset"):
            query = st.text_input("Search data", key="global_search", placeholder="Product, region, retailer...")
            if query:
                if filtered_df is None:
                    st.info("Connect a dataset to search its records.")
                else:
                    searchable = filtered_df.astype("string")
                    matches = searchable.apply(
                        lambda column: column.str.contains(re.escape(query), case=False, na=False)
                    ).any(axis=1)
                    results = filtered_df.loc[matches]
                    st.caption(f"{len(results):,} matching records")
                    if results.empty:
                        st.info("No records match this search.")
                    else:
                        st.dataframe(results.head(100), hide_index=True, width="stretch")
                        if len(results) > 100:
                            st.caption("Showing the first 100 matches.")
    with status_col:
        live_col, filter_col, theme_col, notification_col = st.columns(
            [1.2, .9, .6, .6], gap="small", vertical_alignment="center"
        )
        with live_col:
            source_status = "Dataset loaded" if raw_df is not None else "Demo data"
            st.markdown(
                f"<div><span class='fp-livepill'><span class='fp-dot'></span> {source_status}</span></div>",
                unsafe_allow_html=True,
            )
        with filter_col:
            with st.popover("▽ Filters", help="Filter the dashboard data"):
                region_column = "market" if raw_df is not None and "market" in raw_df.columns else "region"
                filter_options = [
                    ("Region", region_column, "dashboard_region", "All regions"),
                    ("State", "state", "dashboard_state", "All states"),
                    ("City", "city", "dashboard_city", "All cities"),
                    ("Product", "product", "dashboard_product", "All products"),
                    ("Retailer", "retailer", "dashboard_retailer", "All retailers"),
                    ("Sales method", "channel", "dashboard_channel", "All channels"),
                ]
                for label, column, key, default in filter_options:
                    values = (
                        sorted(raw_df[column].dropna().astype(str).unique().tolist())
                        if raw_df is not None and column in raw_df.columns else []
                    )
                    options = [default, *values]
                    if st.session_state[key] not in options:
                        st.session_state[key] = default
                    st.selectbox(label, options, key=key)
                if filtered_df is not None:
                    st.caption(f"{len(filtered_df):,} matching records")
                else:
                    st.caption("No dataset loaded")
                st.button("Reset filters", key="reset-dashboard-filters", on_click=reset_dashboard_filters)
        with theme_col:
            theme_icon = ":material/dark_mode:" if st.session_state.dashboard_light_mode else ":material/light_mode:"
            st.button(theme_icon, key="theme-toggle", on_click=toggle_dashboard_theme,
                      help="Switch between light and dark mode", width="content")
        with notification_col:
            if st.button(":material/notifications:", key="notification-trigger", help="Notifications", width="content"):
                weakest = stats.get("low_products", [{}])[0]
                render_notifications(
                    raw_df is not None,
                    source_label,
                    stats["data_rows"],
                    stats["date_range"],
                    weakest.get("name"),
                    weakest.get("value"),
                    st.session_state.settings_alerts,
                )

    st.markdown(f"<div style='border-bottom:1px solid {CARD_BORDER};'></div>", unsafe_allow_html=True)


@st.dialog("Notifications", width="small", icon=":material/notifications:")
def render_notifications(is_connected, source_label, record_count, date_range,
                         weakest_product, weakest_sales, alerts_enabled):
    if is_connected:
        st.success(f"Dataset loaded: {source_label}")
        st.caption(f"{record_count:,} records match the active filters · {date_range}")
    else:
        st.info("Using built-in demo data. Connect a dataset from Settings to see live records.")

    if weakest_product:
        st.warning(f"Review low-sales product: {weakest_product} ({weakest_sales:,.1f}K sales).")
    if alerts_enabled:
        st.caption("Sales alerts are enabled in Settings.")
    else:
        st.caption("Sales alerts are off. Enable them in Settings to receive alert notices.")


def render_sidebar():
    with st.container(key="dashboard-sidebar-panel"):
        st.markdown(
            f"""
            <div class="fp-logo" style="margin-bottom:22px;">
                <div class="fp-logo-sq">✦</div>
                <div class="fp-logo-text"><b>Adidas Foot print</b><span>BUSINESS INTELLIGENCE</span></div>
            </div>
            <div class="fp-side-label">WORKSPACE</div>
            """,
            unsafe_allow_html=True,
        )

        for label, icon in NAV_ITEMS:
            slug = label.lower()
            active = st.session_state.dash_tab == label
            key = f"navA-{slug}" if active else f"navI-{slug}"
            with st.container(key=key):
                st.button(f"{icon}\u2003{label}", key=f"btn-{slug}",
                          on_click=set_tab, args=(label,), width="stretch")

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    return None


# ----------------------------------------------------------------------
# LANDING PAGE
# ----------------------------------------------------------------------
def render_landing(stats):
    st.markdown(
        f"""
        <div class="fp-topbar">
            <div class="fp-logo">
                <svg width="26" height="20" viewBox="0 0 26 20" fill="none">
                    <path d="M0 20L6 8H12L6 20H0Z" fill="white"/>
                    <path d="M9 20L15 8H21L15 20H9Z" fill="white"/>
                    <path d="M18 12L21 6H26L23 12H18Z" fill="{GREEN}"/>
                </svg>
                <b style="font-weight:900; letter-spacing:0.5px; font-size:15px;">ADIDAS FOOT PRINT</b>
            </div>
            <div class="fp-badge"><span class="fp-dot"></span> SALES INTELLIGENCE PLATFORM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hero_left, hero_right = st.columns([1.05, 1], gap="large")

    with hero_left:
        st.markdown(
            f"""
            <div class="fp-eyebrow">
                <svg width="18" height="14" viewBox="0 0 26 20" fill="none">
                    <path d="M0 20L6 8H12L6 20H0Z" fill="{GREEN}"/>
                    <path d="M9 20L15 8H21L15 20H9Z" fill="{GREEN}"/>
                </svg>
                BUILT FOR THE NEXT STEP
            </div>
            <div class="fp-headline">
                EVERY SALE<br>LEAVES A<br><span class="accent">FOOT PRINT.</span>
            </div>
            <div class="fp-sub">
                Adidas Foot print transforms historical sales into clear business
                intelligence—revealing demand trends, forecasting future
                performance, and showing which products move the business forward.
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([1, 2.4])
        with c1:
            st.button("Get Started  ›", key="cta-btn", on_click=go_dashboard, width="stretch")
        with c2:
            st.markdown('<div class="fp-cta-note"></div>', unsafe_allow_html=True)

    with hero_right:
        growth_sign = "+" if stats["growth_pct"] >= 0 else ""
        values = stats["bar_values"]
        months = stats.get("bar_months", [f"Period {index + 1}" for index in range(len(values))])
        hi = stats["bar_highlight_from"]

        st.markdown('<div class="fp-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="fp-card-label">Annualized sales run rate</div>
            <div class="fp-card-row">
                <div class="fp-card-value">${stats['projected_sales']}M</div>
                <div class="fp-pill">{growth_sign}{stats['growth_pct']}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.altair_chart(landing_sales_chart(values, months, hi), width="stretch")

        s1, s2, s3 = st.columns(3)
        for col, label, val in zip(
            (s1, s2, s3),
            ("Products", "Markets", "Trend R²"),
            (stats["products"], f"{stats['markets']:02d}", f"{stats['confidence']}%"),
        ):
            with col:
                st.markdown(
                    f'<div class="fp-stat-box"><div class="fp-stat-label">{label}</div>'
                    f'<div class="fp-stat-value">{val}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    t_left, t_right = st.columns([2, 1])
    with t_left:
        st.markdown('<div class="fp-section-eyebrow">THE TEAM</div>', unsafe_allow_html=True)
        st.markdown('<div class="fp-section-title">Meet the creators</div>', unsafe_allow_html=True)
    with t_right:
        st.markdown(
            '<div style="text-align:right; margin-top:28px;" class="fp-section-note">'
            "Four minds. One smarter dashboard.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(4)
    for col, member in zip(cols, TEAM):
        image_path = Path(__file__).with_name(member["image"])
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        with col:
            st.markdown(
                f"""
                <div class="fp-team-card">
                    <div class="fp-avatar" style="overflow:hidden; padding:0;">
                        <img src="data:{mime};base64,{encoded}" style="width:100%; height:100%; object-fit:cover; border-radius:10px;" />
                    </div>
                    <div>
                        <div class="fp-team-name">{member['name']}</div>
                        <div class="fp-team-role">{member['role']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("<br><br>", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# KPI CARD
# ----------------------------------------------------------------------
def kpi_card(icon, icon_bg, icon_color, label, value, delta, note="vs last period",
             delta_suffix="%"):
    delta_markup = ""
    if delta is not None:
        up = delta >= 0
        pill_class = "up" if up else "down"
        arrow = "▲" if up else "▼"
        delta_markup = f'<div class="kpi-pill {pill_class}">{arrow} {abs(delta)}{delta_suffix}</div>'
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-icon" style="background:{icon_bg}; color:{icon_color};">{icon}</div>
                {delta_markup}
            </div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note or ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def revenue_traffic_row(stats):
    c1, c2 = st.columns([1.55, 1], gap="medium")
    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="chart-head">
                <div><p class="chart-title">Revenue overview</p>
                <div class="chart-sub">Monthly revenue and {stats.get('expenses_label', 'estimated expenses').lower()}</div></div>
                <div style="white-space:nowrap;">
                    <span class="legend-dot"><span class="sw" style="background:{GREEN};"></span>Revenue</span>
                    <span class="legend-dot"><span class="sw" style="background:{LAVENDER};"></span>Expenses</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.altair_chart(
            revenue_bar_chart(stats["months"], stats["revenue_series"], stats["expenses_series"]),
            width="stretch",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="chart-head">
                <div><p class="chart-title">Sales method mix</p>
                <div class="chart-sub">Share of sales records by method</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.altair_chart(
            traffic_donut(
                stats["traffic_labels"], stats["traffic_values"],
                stats["traffic_colors"], stats["traffic_total"],
            ),
            width="stretch",
        )
        for label, val, color in zip(stats["traffic_labels"], stats["traffic_values"], stats["traffic_colors"]):
            st.markdown(
                f"""
                <div class="legend-list-item">
                    <span class="left"><span class="sw" style="background:{color};"></span>{label}</span>
                    <span class="pct">{val}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# OVERVIEW PAGE
# ----------------------------------------------------------------------
def render_data_footer(stats):
    if stats.get("data_rows"):
        st.caption(f"Calculated from {stats['data_rows']:,} loaded sales records · {stats['date_range']}.")
    else:
        st.caption("Illustrative demo data; no sales dataset is loaded.")


def build_report_csv(stats):
    rows = [
        {"Section": "KPI", "Metric": "Total revenue", "Period": stats["date_range"],
         "Value": stats["kpi_revenue"], "Unit": "USD millions"},
        {"Section": "KPI", "Metric": stats.get("kpi_profit_label", "Net profit"), "Period": stats["date_range"],
         "Value": stats["kpi_profit"], "Unit": f"USD {stats['kpi_profit_suffix']}"},
        {"Section": "KPI", "Metric": stats.get("kpi_orders_label", "Total orders"),
         "Period": stats["date_range"], "Value": stats["kpi_orders"],
         "Unit": "units" if stats.get("kpi_orders_label") == "Units sold" else "records" if stats.get("kpi_orders_label") == "Sales records" else "orders"},
        {"Section": "KPI", "Metric": stats["kpi_conv_label"], "Period": stats["date_range"],
         "Value": stats["kpi_conv"], "Unit": "%"},
    ]
    rows.extend(
        {"Section": "Monthly", "Metric": "Revenue", "Period": month,
         "Value": revenue, "Unit": "USD thousands"}
        for month, revenue in zip(stats["months"], stats["revenue_series"])
    )
    rows.extend(
        {"Section": "Monthly", "Metric": stats.get("expenses_label", "Estimated expenses"), "Period": month,
         "Value": expense, "Unit": "USD thousands"}
        for month, expense in zip(stats["months"], stats["expenses_series"])
    )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


def render_overview(stats):
    top_l, top_r = st.columns([2.4, 1.2])
    with top_l:
        st.markdown('<div class="fp-dash-title">Executive overview</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="fp-dash-sub">Monitor performance, uncover trends, and make confident decisions.</div>',
            unsafe_allow_html=True,
        )
    with top_r:
        b1, b2 = st.columns([1, 1], vertical_alignment="bottom")
        with b1:
            date_range = render_section_date_range("Overview")
        with b2:
            st.download_button(
                "⭳ Export report", data=build_report_csv(stats),
                file_name="adidas-sales-report.csv", mime="text/csv",
                key="export-btn", width="stretch",
            )

    st.caption(
        f"Filters: {st.session_state.dashboard_region} · {st.session_state.dashboard_channel} · "
        f"{date_range}"
    )
    st.markdown(
        f"""
        <div class="data-scope">
            <div class="data-scope-item"><div class="data-scope-label">Sales records</div>
            <div class="data-scope-value">{stats['data_rows']:,} records</div></div>
            <div class="data-scope-item"><div class="data-scope-label">Historical period</div>
            <div class="data-scope-value">{stats['date_range']}</div></div>
            <div class="data-scope-item"><div class="data-scope-label">Products tracked</div>
            <div class="data-scope-value">{stats['products']}</div></div>
            <div class="data-scope-item"><div class="data-scope-label">Regions tracked</div>
            <div class="data-scope-value">{stats['region_count']}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    delta_note = stats.get("kpi_delta_note", "vs last period")
    with k1:
        kpi_card("💲", "rgba(163,230,53,0.12)", GREEN, "Total revenue",
                  f"${stats['kpi_revenue']}M", stats["kpi_revenue_delta"], note=delta_note)
    with k2:
        kpi_card("📊", "rgba(45,212,191,0.12)", TEAL, stats.get("kpi_profit_label", "Net profit"),
              f"${stats['kpi_profit']}{stats['kpi_profit_suffix']}", stats["kpi_profit_delta"], note=delta_note)
    with k3:
        kpi_card("🛍️", "rgba(96,165,250,0.12)", BLUE,
                 stats.get("kpi_orders_label", "Total orders"),
                 f"{stats['kpi_orders']:,}", stats["kpi_orders_delta"], note=delta_note)
    with k4:
        kpi_card("🛒", "rgba(245,158,11,0.12)", ORANGE, stats["kpi_conv_label"],
                 f"{stats['kpi_conv']}%" if stats["kpi_conv"] is not None else "N/A",
                 stats["kpi_conv_delta"], note=delta_note,
                 delta_suffix=stats.get("kpi_conv_delta_suffix", "%"))

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    revenue_traffic_row(stats)


# ----------------------------------------------------------------------
# ANALYTICS PAGE
# ----------------------------------------------------------------------
def render_analytics(stats):
    title_col, range_col = st.columns([2.5, 1], vertical_alignment="bottom")
    with title_col:
        st.markdown('<div class="fp-dash-title">Analytics</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="fp-dash-sub">Analyze historical trends, seasonal demand, sales channels, and product performance.</div>',
            unsafe_allow_html=True,
        )
    with range_col:
        render_section_date_range("Analytics")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    revenue_traffic_row(stats)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.55, 1], gap="medium")
    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="chart-head">
                <div><p class="chart-title">Business performance</p>
                <div class="chart-sub">Revenue and net profit trend</div></div>
                <div style="white-space:nowrap;">
                    <span class="legend-dot"><span class="sw" style="background:{GREEN};"></span>Revenue</span>
                    <span class="legend-dot"><span class="sw" style="background:{TEAL};"></span>Profit</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig = business_perf_chart(stats["months"], stats["perf_revenue"], stats["perf_profit"])
        st.altair_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        h1, h2 = st.columns([2, 1])
        with h1:
            st.markdown(
                """<p class="chart-title">Top products</p>
                <div class="chart-sub">Highest revenue contributors</div>""",
                unsafe_allow_html=True,
            )
        with h2:
            st.markdown('<div style="text-align:right; padding-top:4px;" class="view-all">View all</div>',
                        unsafe_allow_html=True)

        max_val = max(p["value"] for p in stats["top_products"]) or 1
        for i, p in enumerate(stats["top_products"], start=1):
            pct = max(6, round(p["value"] / max_val * 100))
            st.markdown(
                f"""
                <div class="prod-row">
                    <div class="prod-top">
                        <div class="prod-left">
                            <div class="rank-badge">{i}</div>
                            <div><div class="prod-name">{p['name']}</div>
                            <div class="prod-cat">{p['category']}</div></div>
                        </div>
                        <div class="prod-value">${p['value']}K</div>
                    </div>
                    <div class="prod-bar-bg"><div class="prod-bar-fill" style="width:{pct}%; background:{p['color']};"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    render_data_footer(stats)


# ----------------------------------------------------------------------
# PRODUCTS PAGE
# ----------------------------------------------------------------------
def render_products(stats):
    st.markdown('<div class="fp-dash-title">Products</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fp-dash-sub">Drill down into product categories, top performers, and products with low sales.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    chart_col, products_col = st.columns([1.55, 1], gap="medium")
    with chart_col:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="chart-head">
                <div><p class="chart-title">Business performance</p>
                <div class="chart-sub">Revenue and net profit trend</div></div>
                <div style="white-space:nowrap;">
                    <span class="legend-dot"><span class="sw" style="background:{GREEN};"></span>Revenue</span>
                    <span class="legend-dot"><span class="sw" style="background:{TEAL};"></span>Profit</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig = business_perf_chart(stats["months"], stats["perf_revenue"], stats["perf_profit"])
        st.altair_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with products_col:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="chart-head"><div><p class="chart-title">Top products</p>'
            '<div class="chart-sub">Highest revenue contributors</div></div>'
            '<div class="view-all">View all</div></div>',
            unsafe_allow_html=True,
        )
        max_val = max((p["value"] for p in stats["top_products"]), default=1) or 1
        for index, product in enumerate(stats["top_products"], start=1):
            percent = max(6, round(product["value"] / max_val * 100))
            st.markdown(
                f"""
                <div class="prod-row">
                    <div class="prod-top">
                        <div class="prod-left"><div class="rank-badge">{index}</div>
                        <div><div class="prod-name">{product['name']}</div>
                        <div class="prod-cat">{product['category']}</div></div></div>
                        <div class="prod-value">${product['value']}K</div>
                    </div>
                    <div class="prod-bar-bg"><div class="prod-bar-fill" style="width:{percent}%; background:{product['color']};"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    if stats.get("low_products"):
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(
            '<p class="chart-title">Low-sales products</p>'
            '<div class="chart-sub">Products that need attention based on total sales</div>',
            unsafe_allow_html=True,
        )
        low_max = max((product["value"] for product in stats["low_products"]), default=1) or 1
        for product in stats["low_products"]:
            percent = max(6, round(product["value"] / low_max * 100))
            st.markdown(
                f"""
                <div class="prod-row">
                    <div class="prod-top"><div class="prod-left">
                        <div class="rank-badge" style="color:{RED};">!</div>
                        <div><div class="prod-name">{product['name']}</div>
                        <div class="prod-cat">{product['category']}</div></div>
                    </div><div class="prod-value">${product['value']}K</div></div>
                    <div class="prod-bar-bg"><div class="prod-bar-fill" style="width:{percent}%; background:{RED};"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    render_data_footer(stats)


# ----------------------------------------------------------------------
# REPORTS PAGE
# ----------------------------------------------------------------------
def render_reports(stats):
    regions = stats["regional_performance"]

    st.markdown('<div class="fp-dash-title">Reports</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fp-dash-sub">Review sales and growth across regions.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chart-card reports-card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="chart-head reports-heading">
            <div><p class="chart-title">Regional performance</p>
            <div class="chart-sub">Regional sales, {stats.get('regional_volume_label', 'orders').lower()}, {stats.get('regional_average_label', 'average order value').lower()}, and year-over-year growth</div></div>
            <div class="view-all">Detailed report&nbsp;›</div>
        </div>
        <div class="reports-table-head">
            <span>REGION</span><span>TOTAL SALES</span>
            <span>{stats.get('regional_volume_label', 'Orders').upper()}</span>
            <span>{stats.get('regional_average_label', 'Average order value').upper()}</span>
            <span>YOY GROWTH</span><span>7-MONTH FORECAST</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for region in regions:
        tone = "growth-positive" if region["positive"] else "growth-negative" if region["positive"] is not None else ""
        arrow = "⌃" if region["positive"] else "⌄" if region["positive"] is not None else ""
        growth_label = f"{arrow}&nbsp; {region['growth']}%" if region["growth"] is not None else "N/A"
        forecast_sales = (
            f"${region['forecast_sales']:,.2f}M"
            if region.get("forecast_sales") is not None else "N/A"
        )
        st.markdown(
            f"""
            <div class="reports-row">
                <div class="reports-region"><span class="region-code">{region['code']}</span><b>{region['name']}</b></div>
                <strong>${region['sales']:,.0f}</strong><span>{region['volume']:,}</span>
                <span>${region['average']:,.2f}</span>
                <span class="growth-pill {tone}">{growth_label}</span>
                <span>{forecast_sales}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    render_data_footer(stats)


# ----------------------------------------------------------------------
# SETTINGS PAGE
# ----------------------------------------------------------------------
def render_settings():
    st.markdown('<div class="fp-dash-title">Settings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fp-dash-sub">Manage the active dataset and in-app alert preference.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(
            '<p class="chart-title">Notifications</p><div class="chart-sub">In-app alert preference</div>',
            unsafe_allow_html=True,
        )
        st.toggle("Enable alerts", value=st.session_state.settings_alerts, key="settings_alerts")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(
            '<p class="chart-title">Data source</p><div class="chart-sub">Upload a CSV or Excel sales dataset</div>',
            unsafe_allow_html=True,
        )
        uploaded_dataset = st.session_state.active_dataset
        workspace_file = Path(__file__).with_name("Adidas US Sales Datasets.xlsx")
        if uploaded_dataset is not None:
            active_name = uploaded_dataset[0]
            source_status = "Uploaded for this session"
        elif workspace_file.exists():
            active_name = workspace_file.name
            source_status = "Workspace file"
        else:
            active_name = "Demo data"
            source_status = "Built-in demo"
        st.file_uploader(
            "Upload a new dataset",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
            key="dataset_upload",
            on_change=persist_uploaded_dataset,
        )
        if uploaded_dataset is not None:
            st.button(
                "Use workspace dataset",
                key="clear-uploaded-dataset",
                on_click=clear_uploaded_dataset,
            )
        st.markdown(
            f"<div style='margin-top:16px; color:#c7c7c7; font-size:13px; line-height:1.6;'>"
            f"<strong>Active source:</strong> {active_name}<br>"
            f"<strong>Status:</strong> {source_status}<br>"
            "Updates when the app reruns; no live data connection is configured.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Alert preference is stored for this session.")


# ----------------------------------------------------------------------
# SIMULATION PAGE ("Sales forecasting lab")
# ----------------------------------------------------------------------
def compute_sim_stats(sim_base: dict) -> dict:
    """Apply the selected scenario to the trained sales forecast."""
    scenario = st.session_state.scenario
    cfg = SCENARIOS[scenario]
    s = {k: (v.copy() if isinstance(v, (list, dict)) else v) for k, v in sim_base.items()}

    base_forecast_growth = sim_base.get("forecast_growth_pct")
    if base_forecast_growth is None:
        base_forecast_growth = sim_base["growth_pct"]
    growth = round(base_forecast_growth * cfg["growth_mult"], 1)
    s["growth_pct"] = growth
    s["band_max"] = round(sim_base["band_max"] * cfg["band_mult"], 1)

    hist = sim_base["hist_series"]
    split_val = hist[-1]
    forecast = [
        max(0, round(split_val + (value - split_val) * cfg["growth_mult"], 1))
        for value in sim_base["forecast_series"]
    ]
    s["projected_sales"] = round(sum(forecast) / 1000, 2)
    s["hist_series"] = hist
    s["forecast_series"] = forecast

    scaled_products = []
    for p in sim_base["product_outlook"]:
        p2 = dict(p)
        if p.get("forecast_series") is not None:
            last_actual = float(p["last_actual"])
            forecast_series = [
                max(0, round(last_actual + (value - last_actual) * cfg["growth_mult"], 1))
                for value in p["forecast_series"]
            ]
            p2["forecast_series"] = forecast_series
            p2["projected_sales"] = round(sum(forecast_series) / 1000, 2)
            baseline = float(p["baseline_sales"])
            p2["forecast"] = (
                round((float(np.mean(forecast_series)) / baseline - 1) * 100, 1)
                if baseline > 0 else None
            )
            if p2["forecast"] is not None:
                p2["status"] = (
                    "Strong momentum" if p2["forecast"] >= 10
                    else "Above trend" if p2["forecast"] >= 0 else "At risk"
                )
        elif p.get("forecast") is not None:
            p2["forecast"] = round(p["forecast"] * cfg["growth_mult"], 1)
        scaled_products.append(p2)
    s["product_outlook"] = scaled_products
    return s


def render_simulation(sim_base):
    if "forecast_model" not in sim_base:
        sim_base = {
            **sim_base,
            **train_sales_forecast(tuple(float(value) for value in sim_base["hist_series"])),
        }
    stats = compute_sim_stats(sim_base)
    scenario = st.session_state.scenario

    head_l, head_r = st.columns([1.1, 1.9], vertical_alignment="center")
    with head_l:
        st.markdown(
            """
            <div class="fp-eyebrow-sm">✦ PREDICTIVE SIMULATION</div>
            <div class="fp-sim-title">Sales forecasting lab</div>
            <div class="fp-sim-sub">Model future sales from historical patterns and compare product momentum.</div>
            """,
            unsafe_allow_html=True,
        )
    with head_r:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        with st.container(key="scenario-wrap"):
            c1, c2, c3 = st.columns(3)
            for col, label in zip((c1, c2, c3), ["Conservative", "Base case", "Accelerated"]):
                slug = label.lower().replace(" ", "-")
                active = scenario == label
                key = f"scenA-{slug}" if active else f"scenI-{slug}"
                with col:
                    with st.container(key=key):
                        st.button(label, key=f"btn-scen-{slug}", on_click=set_scenario,
                                  args=(label,), width="stretch")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.55, 1], gap="medium")
    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="chart-head">
                <div><p class="chart-title">Historical trend &amp; sales forecast</p>
                <div class="chart-sub">7-month forecast from the trained {stats['forecast_model']} model</div></div>
                <div class="fp-base-pill">{scenario}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(
                f"""<div class="sim-stat-box"><div class="sim-stat-label">7-month sales</div>
                <div class="sim-stat-value">${stats['projected_sales']}M</div></div>""",
                unsafe_allow_html=True,
            )
        with s2:
            sign = "+" if stats["growth_pct"] >= 0 else ""
            st.markdown(
                f"""<div class="sim-stat-box"><div class="sim-stat-label">Forecast vs recent average</div>
                <div class="sim-stat-value pos">{sign}{stats['growth_pct']}%</div></div>""",
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                f"""<div class="sim-stat-box"><div class="sim-stat-label">In-sample fit</div>
                <div class="sim-stat-value">{stats['model_fit']}%</div></div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        fig = simulation_chart(
            stats["hist_series"], stats["forecast_series"], stats["band_max"],
            sim_base.get("hist_months"), sim_base.get("forecast_months"),
        )
        st.altair_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        h1, h2 = st.columns([1.7, 1])
        with h1:
            st.markdown(
                """<p class="chart-title">Product outlook</p>
                <div class="chart-sub">Forecast growth and performance signals</div>""",
                unsafe_allow_html=True,
            )
        with h2:
            st.markdown('<div style="text-align:right; padding-top:2px;" class="outlook-head-note">NEXT 7 MONTHS</div>',
                        unsafe_allow_html=True)

        for p in stats["product_outlook"]:
            forecast_change = p.get("forecast")
            fpos = forecast_change is not None and forecast_change >= 0
            fclass = "pos" if fpos else "neg" if forecast_change is not None else "neutral"
            fsign = "+" if fpos else ""
            fval = (
                "Insufficient history" if forecast_change is None
                else int(forecast_change) if float(forecast_change).is_integer()
                else forecast_change
            )
            forecast_value = (
                f"${p['projected_sales']:,.2f}M projected"
                if p.get("projected_sales") is not None else f"${p['value']}K"
            )
            st.markdown(
                f"""
                <div class="outlook-row">
                    <div class="outlook-top">
                        <div><div class="outlook-name">{p['name']}</div>
                        <div class="outlook-status" style="color:{p['color']};">{p['status']}</div></div>
                        <div>
                            <div class="outlook-value">{forecast_value}</div>
                            <div class="outlook-forecast {fclass}">{fsign}{fval}{'%' if forecast_change is not None else ''} vs trailing average</div>
                        </div>
                    </div>
                    <div class="prod-bar-bg"><div class="prod-bar-fill" style="width:{p['bar_pct']}%; background:{p['color']};"></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        insight_text = (
            "Each product category is forecast from its own monthly sales history."
            if sim_base.get("data_rows") else
            f"Demo outlook: {sim_base['insight_product']} represents {sim_base['insight_pct']}% of sample sales."
        )
        st.markdown(
            f"""<div class="insight-box"><span>Model note:</span><span>{insight_text}</span></div>""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    render_data_footer(stats)


def render_dashboard_content(raw_df, filtered_df, stats, source_label):
    with st.container(key="dashboard-sticky-header"):
        render_topbar(raw_df, filtered_df, stats, source_label)
    sidebar_label = "<<" if st.session_state.sidebar_visible else ">>"
    sidebar_help = "Close sidebar" if st.session_state.sidebar_visible else "Open sidebar"
    st.button(
        sidebar_label,
        key="sidebar-toggle",
        on_click=toggle_sidebar,
        help=sidebar_help,
        width="content",
    )

    if filtered_df is not None and filtered_df.empty:
        st.warning("No records match the active filters. Adjust the date range, region, or sales channel.")
    elif st.session_state.dash_tab == "Overview":
        render_overview(stats)
    elif st.session_state.dash_tab == "Analytics":
        render_analytics(stats)
    elif st.session_state.dash_tab == "Simulation":
        if raw_df is not None and "hist_series" not in stats:
            st.info("The active filters leave less than two months of sales history, which is not enough to train a forecast. Expand the date range or reset filters.")
        else:
            render_simulation(stats if raw_df is not None else SIM_DEFAULTS)
    elif st.session_state.dash_tab == "Products":
        render_products(stats)
    elif st.session_state.dash_tab == "Reports":
        render_reports(stats)
    elif st.session_state.dash_tab == "Settings":
        render_settings()
    else:
        st.markdown(f'<div class="fp-dash-title">{st.session_state.dash_tab}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="fp-dash-sub">The {st.session_state.dash_tab} section isn\'t part of the '
            "provided mockups yet — Overview and Analytics are fully built out. "
            "Let me know if you'd like this section designed too.</div>",
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------
# ROUTING
# ----------------------------------------------------------------------
workspace_dataset = Path(__file__).with_name("Adidas US Sales Datasets.xlsx")
active_upload = st.session_state.active_dataset
data_source = (
    active_upload
    if active_upload is not None
    else workspace_dataset if workspace_dataset.exists() else None
)
source_label = (
    active_upload[0]
    if active_upload is not None
    else data_source.name if data_source is not None else "Demo data"
)
raw_df = load_dataframe(data_source) if data_source is not None else None

if st.session_state.page == "landing":
    landing_stats = compute_stats(raw_df) if raw_df is not None else DEFAULTS
    render_landing(landing_stats)
else:
    active_section = st.session_state.dash_tab
    active_date_range = (
        st.session_state.section_date_ranges.get(active_section, "Last 12 months")
        if active_section in DATE_RANGE_SECTIONS else "All time"
    )
    filtered_df = (
        filter_dashboard_dataframe(raw_df, active_date_range)
        if raw_df is not None else None
    )
    if filtered_df is not None and filtered_df.empty:
        stats = dict(DEFAULTS)
        stats.update({
            "data_rows": 0,
            "date_range": "No matching records",
            "products": 0,
            "markets": 0,
            "region_count": 0,
            "kpi_revenue": 0,
            "kpi_revenue_delta": None,
            "kpi_profit": 0,
            "kpi_profit_delta": None,
            "kpi_profit_label": "Net profit",
            "kpi_orders": 0,
            "kpi_orders_delta": None,
            "kpi_orders_label": "Sales records",
            "kpi_conv": None,
            "kpi_conv_delta": None,
            "kpi_conv_label": "Conversion unavailable",
            "kpi_delta_note": None,
            "months": [],
            "revenue_series": [],
            "expenses_series": [],
            "perf_revenue": [],
            "perf_profit": [],
            "traffic_labels": [],
            "traffic_values": [],
            "traffic_colors": [],
            "traffic_total": "0 records",
            "top_products": [],
            "low_products": [],
            "regional_performance": [],
        })
    else:
        stats = compute_stats(filtered_df) if filtered_df is not None else dict(DEFAULTS)

    with st.container(key="dashboard-layout-shell"):
        if st.session_state.sidebar_visible:
            sidebar_column, content_column = st.columns([1.2, 4.8], gap="small", vertical_alignment="top")
            with sidebar_column:
                render_sidebar()
            with content_column:
                render_dashboard_content(raw_df, filtered_df, stats, source_label)
        else:
            render_dashboard_content(raw_df, filtered_df, stats, source_label)