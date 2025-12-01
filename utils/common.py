# ================================================================
# utils/common.py — CyberGlass Aqua helpers
# ================================================================
from __future__ import annotations
import pathlib
import logging
import streamlit as st

# Root ialah folder .../dashboard
ROOT = pathlib.Path(__file__).resolve().parents[1]

def get_paths() -> dict[str, pathlib.Path]:
    """
    Pulangkan lokasi penting.
    - DATA     : ../data
    - OUTPUT   : ../output  (dataset final & model final di sini)
    - MODELS   : alias ke OUTPUT (supaya mudah)
    - ASSETS   : ./assets (cth: geojson)
    """
    data_dir   = ROOT.parent / "data"
    output_dir = ROOT.parent / "output"
    assets_dir = ROOT / "assets"
    for p in (data_dir, output_dir, assets_dir):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "ROOT": ROOT,
        "DATA": data_dir,
        "OUTPUT": output_dir,
        "MODELS": output_dir,
        "ASSETS": assets_dir,
    }

def theme_css() -> str:
    """CyberGlass Aqua theme (tanpa depend luar)."""
    return """
    <style>
      body {
        background: radial-gradient(circle at 10% 20%, #0d1b2a, #000814 90%) !important;
        color: #e0f7fa !important;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      }
      h1,h2,h3 { color:#90e0ef; font-weight:700; }
      .cy-divider { height:2px; background:linear-gradient(90deg,transparent,#00b4d8,transparent);
                    margin:25px 0; border-radius:2px; }
      .cy-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(0,255,255,0.12);
        border-radius: 16px;
        backdrop-filter: blur(10px);
        box-shadow: 0 0 18px rgba(0,180,216,0.18);
        padding: 18px;
      }
      .metric-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(0,255,255,0.12);
        border-radius: 16px;
        padding: 18px; text-align:center;
        box-shadow: 0 0 15px rgba(0,180,216,0.2);
        transition: .25s ease; backdrop-filter: blur(10px);
      }
      .metric-box:hover { transform: translateY(-4px);
        box-shadow: 0 0 25px rgba(0,229,255,0.5); }
      .metric-value { font-size:28px; color:#48cae4; font-weight:700; }
      .metric-label { color:#ade8f4; font-size:13px; letter-spacing:.5px; text-transform:uppercase; }
    </style>
    """

# simple logger (Streamlit-safe)
_logger = None
def log() -> logging.Logger:
    global _logger
    if _logger: return _logger
    _logger = logging.getLogger("crime_dash")
    if not _logger.handlers:
        _logger.setLevel(logging.INFO)
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        _logger.addHandler(_h)
    return _logger

def status_badge(text:str, kind:str="info"):
    """Lencana status kecil ala pill."""
    color = {"ok":"#06d6a0","warn":"#ffd166","info":"#00e0ff","err":"#ef476f"}.get(kind,"#00e0ff")
    st.markdown(
        f"<span style='display:inline-block;padding:6px 10px;border-radius:999px;"
        f"background:{color}22;border:1px solid {color}55;color:{color};font-size:12px;'>"
        f"{text}</span>", unsafe_allow_html=True
    )
