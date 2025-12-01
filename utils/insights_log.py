# ================================================================
# utils/insights_log.py — v15.0 Pro
# Auto Insights Log (JSONL) + quick history view
# ================================================================
from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd
import streamlit as st

def _log_path(out_dir: Path) -> Path:
    p = Path(out_dir) / "insights_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("", encoding="utf-8")
    return p

def log_event(event_type: str, details: dict):
    try:
        out_dir = st.session_state.get("OUTPUT_DIR")
        if out_dir is None:
            # recover from common helper
            from utils.common import get_paths
            out_dir = get_paths()["OUTPUT"]
        path = _log_path(out_dir)
        rec = {"ts": int(time.time()), "event": event_type, "details": details}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass

def list_history(out_dir: Path):
    p = _log_path(out_dir)
    rows = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            rows.append(json.loads(line))
    except Exception:
        pass
    return rows

def render_history_table(records: list[dict]):
    if not records:
        st.info("No history yet — run some analyses to populate logs.")
        return
    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df["ts"], unit="s")
    st.dataframe(df[["time", "event", "details"]].sort_values("time", ascending=False), use_container_width=True)
