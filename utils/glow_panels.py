# ===============================================================
# 🌌 utils/glow_panels.py — Animated Aqua Glow Panels for Charts
# ===============================================================
import streamlit as st

def add_glow_effect():
    """
    Injects glowing animated background layers behind all cards, charts, and dataframes.
    This matches the CyberGlass Aqua theme perfectly.
    """
    st.markdown("""
    <style>
    /* =======================================================
       🌊 GLOBAL AQUA-GLOW PANEL EFFECT
    ======================================================= */
    .glow-card, .stDataFrame, .stPlotlyChart, .stMetric, .css-1dp5vir, .block-container {
        position: relative !important;
        z-index: 1 !important;
    }

    /* Animated background gradient for charts & cards */
    .glow-card::before,
    .stDataFrame::before,
    .stPlotlyChart::before,
    .stMetric::before {
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg,
            rgba(0,255,255,0.12),
            rgba(0,180,255,0.06),
            rgba(0,100,150,0.08)
        );
        border-radius: 20px;
        filter: blur(12px);
        animation: aquaGlowMove 8s ease-in-out infinite alternate;
        z-index: -1;
    }

    @keyframes aquaGlowMove {
        0% { opacity: 0.6; transform: translate(0px, 0px); }
        50% { opacity: 1; transform: translate(8px, -8px) scale(1.03); }
        100% { opacity: 0.6; transform: translate(-6px, 6px) scale(1.0); }
    }

    /* Metric & Card Styling */
    .stMetric {
        background: rgba(0, 40, 60, 0.4);
        border: 1px solid rgba(0, 255, 255, 0.15);
        border-radius: 16px;
        padding: 10px 16px;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.1);
        transition: all 0.4s ease;
    }
    .stMetric:hover {
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.3);
        transform: translateY(-2px);
    }

    /* DataFrames with glowing edges */
    .stDataFrame {
        border: 1px solid rgba(0, 255, 255, 0.15);
        border-radius: 10px;
        background: rgba(0, 30, 40, 0.5);
        box-shadow: inset 0 0 20px rgba(0,255,255,0.05);
        padding: 6px;
    }

    /* Plotly Charts embedded in glowing cards */
    .stPlotlyChart {
        border-radius: 20px;
        background: rgba(0, 50, 70, 0.35);
        border: 1px solid rgba(0,255,255,0.15);
        box-shadow: 0 0 25px rgba(0,255,255,0.15);
        padding: 8px;
        transition: 0.4s ease;
    }
    .stPlotlyChart:hover {
        box-shadow: 0 0 35px rgba(0,255,255,0.3);
        transform: scale(1.01);
    }

    /* Headings Glow */
    h1, h2, h3, h4 {
        color: #bde0fe !important;
        text-shadow: 0 0 10px rgba(0,255,255,0.4);
    }

    /* Buttons styled to match */
    div[data-testid="stButton"] > button {
        background: linear-gradient(90deg, rgba(0,255,255,0.25), rgba(0,150,255,0.35));
        color: #e0faff !important;
        border: 1px solid rgba(0,255,255,0.25);
        border-radius: 12px;
        padding: 6px 16px;
        font-weight: 600;
        text-shadow: 0 0 8px rgba(0,255,255,0.4);
        transition: 0.3s ease;
    }
    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(90deg, rgba(0,200,255,0.35), rgba(0,255,255,0.45));
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(0,255,255,0.4);
    }

    /* Inputs and Selectboxes Glow */
    div[data-testid="stSelectbox"], div[data-testid="stTextInput"], div[data-testid="stNumberInput"] {
        background: rgba(0, 50, 70, 0.35);
        border: 1px solid rgba(0,255,255,0.2);
        border-radius: 10px;
        box-shadow: inset 0 0 20px rgba(0,255,255,0.05);
    }
    </style>
    """, unsafe_allow_html=True)
