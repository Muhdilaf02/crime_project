import streamlit as st

def show_banner(title: str, steps: list[str]) -> None:
    """Always-visible guidance banner with modern styling."""
    st.markdown(
        """
        <style>
        .guide-card{
            background: linear-gradient(135deg, rgba(16,24,39,.6), rgba(8,47,73,.5));
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 14px;
            padding: 14px 16px;
            backdrop-filter: blur(14px);
            box-shadow: 0 8px 24px rgba(0,0,0,.25) inset, 0 8px 20px rgba(0,0,0,.15);
        }
        .guide-title{
            font-weight: 800; letter-spacing:.3px; margin:0 0 6px 0;
            background: linear-gradient(90deg,#67e8f9,#a78bfa,#f472b6);
            -webkit-background-clip:text; background-clip:text; color:transparent;
        }
        .guide-step{ margin: 2px 0; }
        .guide-step b{ color:#e2e8f0; }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown('<div class="guide-card">', unsafe_allow_html=True)
    st.markdown(f'<h4 class="guide-title">🧭 {title}</h4>', unsafe_allow_html=True)
    for i, step in enumerate(steps, 1):
        st.markdown(f'<div class="guide-step">{i}. <b>{step}</b></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
