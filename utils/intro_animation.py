# ===============================================================
# 🌌 utils/intro_animation.py — Clean & Smooth Cyber Intro
# ===============================================================
import streamlit as st
import time
import base64
import os

def show_intro():

    if "intro_shown" not in st.session_state:
        st.session_state["intro_shown"] = True

        # Correct logo path
        logo_path = os.path.join(os.path.dirname(__file__), "../../output/LOGO.png")

        # Convert PNG → Base64
        with open(logo_path, "rb") as img_file:
            logo_base64 = base64.b64encode(img_file.read()).decode()

        # Inject HTML/CSS
        st.markdown(f"""
        <style>

        /* =========================================================
           🌌 CLEAN, SMOOTH OVERLAY
        ==========================================================*/
        .intro-overlay {{
            position: fixed;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;

            background: radial-gradient(circle at center,
                rgba(0,40,60,1) 0%,
                rgba(0,0,10,0.96) 100%
            );

            z-index: 9999999;

            /* Smooth 3.2s fade-out after everything appears */
            animation: fadeOut 3.2s ease-in-out forwards;
            animation-delay: 2.9s;
        }}

        @keyframes fadeOut {{
            0% {{ opacity: 1; }}
            75% {{ opacity: 0.85; }}
            100% {{ opacity: 0; visibility: hidden; }}
        }}

        /* =========================================================
           ⭐ LOGO (Smooth + Clean)
        ==========================================================*/
        .intro-logo {{
            width: 360px;
            opacity: 0;
            animation: logoIn 1.4s ease-out forwards;
            animation-delay: 0.10s;
            filter: drop-shadow(0 0 14px #00e0ff);
        }}

        @keyframes logoIn {{
            0% {{ opacity: 0; transform: scale(0.92); }}
            100% {{ opacity: 1; transform: scale(1); }}
        }}

        /* =========================================================
           ✨ MAIN TITLE — Smooth Fade + Gentle Lift
        ==========================================================*/
        .intro-text {{
            font-size: 40px;
            margin-top: 12px;
            font-weight: 700;
            letter-spacing: 2px;
            color: #00e0ff;
            font-family: 'Orbitron', sans-serif;

            opacity: 0;
            animation: textIn 1.4s ease-out forwards;
            animation-delay: 0.55s;

            text-shadow:
                0 0 12px #00ffff,
                0 0 40px rgba(0,255,255,0.5);
        }}

        @keyframes textIn {{
            0% {{ opacity: 0; transform: translateY(14px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}

        /* =========================================================
           🔹 SUBTITLE — Ultra Smooth Fade
        ==========================================================*/
        .intro-sub {{
            font-size: 17px;
            margin-top: 6px;
            color: #bde0fe;
            font-family: 'Roboto Mono', monospace;

            opacity: 0;
            animation: subIn 1.5s ease-out forwards;
            animation-delay: 1.1s;

            text-shadow: 0 0 10px rgba(0,255,255,0.35);
        }}

        @keyframes subIn {{
            0% {{ opacity: 0; transform: translateY(12px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}

        </style>


        <!-- =========================================================
             ⭐ HTML CONTENT
        ==========================================================-->
        <div class="intro-overlay">
            <img class="intro-logo" src="data:image/png;base64,{logo_base64}">
            <div class="intro-text">WELCOME CRIMELENS INDIA DASHBOARD</div>
            <div class="intro-sub">EXPLORING CRIME & SOCIOECONOMIC CORRELATION</div>
        </div>

        """, unsafe_allow_html=True)

        time.sleep(3.3)
