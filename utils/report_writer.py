# ================================================================
# utils/report_writer.py — v15.0 Pro
# AI Story Mode (deterministic narrative + exportable report)
# ================================================================
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import datetime as dt

def _summarize(df: pd.DataFrame, crime_cols, socio_cols, region=None, year=None):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not crime_cols:
        df["total_crimes_reported"] = 1
        crime_cols = ["total_crimes_reported"]
    crime_avg = df[crime_cols].mean(numeric_only=True).mean()
    s_corr = df[num_cols].corr().mean().mean()

    parts = []
    title = f"Insight Brief — {region or 'All Regions'} ({year or 'All Years'})"
    parts.append(f"# {title}\n")
    parts.append(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    parts.append("## Executive Highlights\n")
    parts.append(f"- Overall crime level (mean of crime indicators): **{crime_avg:,.2f}**")
    parts.append(f"- Average numeric correlation strength (global): **{s_corr:.2f}**")

    if region:
        # regional deviations
        region_col = "district" if "district" in df.columns and region in set(df["district"].astype(str)) else ("state" if "state" in df.columns else "state_name")
        rdf = df[df[region_col].astype(str) == str(region)]
        if len(rdf):
            avg_nat = df[socio_cols].mean(numeric_only=True)
            avg_reg = rdf[socio_cols].mean(numeric_only=True)
            cmp = ((avg_reg - avg_nat) / avg_nat * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
            top_pos = ", ".join(cmp.sort_values(ascending=False).head(3).index.tolist())
            top_neg = ", ".join(cmp.sort_values().head(3).index.tolist())
            parts.append("\n## Regional Snapshot\n")
            parts.append(f"- Region: **{region}**")
            parts.append(f"- Top strengths: **{top_pos or '—'}**")
            parts.append(f"- Areas for improvement: **{top_neg or '—'}**")
    parts.append("\n## Strategic Note\n- Prioritize targeted social programs where indicators lag.\n- Align patrol/resource allocation with data-driven hotspots.\n")
    return "\n".join(parts)

def render_story_mode(df: pd.DataFrame, crime_cols, socio_cols, auto_region, auto_year, out_dir: Path):
    st.subheader("📄 Story Mode — Auto Report Writer")
    st.caption("One-click executive summary based on your dataset & (optionally) the last predicted region/year.")

    # Inputs
    region = st.text_input("Region focus (optional)", value=str(auto_region) if auto_region else "")
    year = st.text_input("Year (optional)", value=str(auto_year) if auto_year else "")

    if st.button("📝 Generate Report", use_container_width=True):
        report = _summarize(df, crime_cols, socio_cols, region=region if region else None, year=year if year else None)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"insight_report_{(region or 'all').replace(' ','_')}_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.md"
        path.write_text(report, encoding="utf-8")
        st.success(f"✅ Report saved: {path}")
        st.download_button("⬇️ Download Report", data=report.encode("utf-8"), file_name=path.name, mime="text/markdown")
        st.markdown("---")
        st.markdown(report)
