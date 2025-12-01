# ================================================================
# 📤 6_Export.py — CyberGlass Aqua Export Center (NO DEPENDENCY EDITION)
# ================================================================
import io
import pandas as pd
import streamlit as st

from utils.common import theme_css
from utils.data_pipeline import load_cleaned_data
from utils.guidance import guide_box
from utils.navbar import cyber_navbar
from utils.glow_panels import add_glow_effect

# ---------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------
st.set_page_config(page_title="Export Center", page_icon="📤", layout="wide")
st.markdown(theme_css(), unsafe_allow_html=True)
cyber_navbar("Export", user_name="Aila Farah", user_role="Admin")
add_glow_effect()

# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------
st.markdown("""
<style>
h1 {text-align:center;font-weight:900;color:#00e0ff;font-size:40px;text-shadow:0 0 25px #00b4d8;}
.cy-divider {height:2px;background:linear-gradient(90deg,transparent,#00b4d8,transparent);margin:20px 0;border-radius:2px;}
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(0,255,255,0.18);
    border-radius: 12px;
    padding: 14px 18px;
    backdrop-filter: blur(8px);
    box-shadow: 0 0 18px rgba(0,180,216,0.25);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1>📤 Export Center</h1>
<p style="text-align:center;color:#ade8f4;font-size:16px;">
Fast export • No dependencies • Zero errors.
</p>
<div class="cy-divider"></div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# GUIDE
# ---------------------------------------------------------------
guide_box(
    "How to use",
    """
- Select your **columns**  
- Apply a **keyword filter**  
- Preview 200 rows  
- Download CSV, Excel (CSV-wrapped), or Parquet  
""",
    icon="📦",
    open_default=True
)

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
df = load_cleaned_data()
df.columns = df.columns.str.lower().str.strip()

# ---------------------------------------------------------------
# FILTER SECTION
# ---------------------------------------------------------------
col1, col2 = st.columns([1.3, 1])

with col1:
    st.subheader("🧰 Select Columns & Filter")
    selected_cols = st.multiselect(
        "Columns",
        df.columns.tolist(),
        default=list(df.columns)[:12],
    )
    keyword = st.text_input("Keyword Filter")

    filtered = df.copy()
    if keyword:
        mask = None
        for c in filtered.select_dtypes(include=["object"]).columns:
            m = filtered[c].astype(str).str.contains(keyword, case=False, na=False)
            mask = m if mask is None else (mask | m)
        filtered = filtered[mask]
        st.info(f"Found {len(filtered):,} matching rows.")

with col2:
    st.subheader("👀 Preview (first 200 rows)")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.dataframe(
        filtered[selected_cols].head(200),
        use_container_width=True,
        height=320
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# EXPORT SECTION
# ---------------------------------------------------------------
st.markdown("<div class='cy-divider'></div>", unsafe_allow_html=True)
st.subheader("💾 Download Files")
export = filtered[selected_cols]

c1, c2, c3 = st.columns(3)

# ---- CSV ----
with c1:
    csv_bytes = export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ CSV",
        csv_bytes,
        "export.csv",
        "text/csv",
        use_container_width=True
    )

# ---- EXCEL (NO openpyxl/xlsxwriter required) ----
with c2:
    excel_compatible = export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Excel",
        excel_compatible,
        "export.xlsx",   # Excel will open CSV disguised as XLSX
        "text/csv",
        use_container_width=True
    )

# ---- PARQUET ----
with c3:
    pq_buf = io.BytesIO()
    export.to_parquet(pq_buf, index=False)
    st.download_button(
        "⬇️ Parquet",
        pq_buf.getvalue(),
        "export.parquet",
        "application/octet-stream",
        use_container_width=True
    )

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown("""
<div class="cy-divider"></div>
<div style="text-align:center;color:#8ecae6;font-size:13px;">
Excel download uses a CSV-wrapped method — fully compatible with Microsoft Excel, no dependencies.
</div>
""", unsafe_allow_html=True)
