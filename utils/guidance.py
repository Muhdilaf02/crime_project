# ================================================================
# utils/guidance.py — user guidance helpers
# ================================================================
import streamlit as st

def guide_box(title: str, content: str, icon: str = "💡", open_default: bool = False):
    with st.expander(f"{icon} {title}", expanded=open_default):
        st.markdown(content)

def steps_box(title: str, steps: list[str], icon: str = "🧭", open_default: bool = True):
    md = "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])
    guide_box(title, md, icon=icon, open_default=open_default)

def chart_explain(title: str, desc: str):
    st.caption(f"**{title}** — {desc}")

def note_other_crime():
    st.info(
        "🔎 **Other Crime**: kategori jenayah yang **tidak termasuk** dalam kumpulan utama "
        "(cth. jenayah ganas, harta benda, siber). Contoh: penipuan dokumen kecil, rasuah kecil, "
        "penyalahgunaan kuasa. Ia berguna sebagai indikator integriti & tatakelola di sesuatu daerah."
    )
