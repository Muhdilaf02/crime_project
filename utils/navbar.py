# ===============================================================
# 🌊 utils/navbar.py — CyberGlass Aqua Navbar + Global Animated Background
# ===============================================================
import streamlit as st

def cyber_navbar(current_page: str, user_name: str = "Guest", user_role: str = "Analyst"):
    """
    Displays a full-screen aqua-glass animated background and floating navbar.
    """

    nav_items = [
        ("🏠 Home", "Home"),
        ("📊 Overview", "1_Overview"),
        ("📈 Analytics", "2_Analytics"),
        ("🗺️ Map", "3_Map"),
        ("🤖 Prediction", "4_Prediction"),
        ("🧭 Intelligence", "5_Crime_Intelligence"),
        ("🧠 Explainability", "6_Explainability"),
        ("📤 Export", "7_Export")
    ]

    # 🌌 Global Background + Navbar Styling
    st.markdown("""
    <style>
    /* =======================================================
       🌌 FULL BACKGROUND PARTICLES (AQUA GLOW)
    ======================================================= */
    body {
        background: radial-gradient(circle at top right, rgba(0, 20, 30, 0.85), rgba(0, 10, 20, 1));
        overflow-x: hidden;
        color: #e0faff;
    }

    .global-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
        z-index: -2;
        background: radial-gradient(circle at 50% 20%, rgba(0, 255, 255, 0.1), rgba(0, 0, 30, 0.95));
    }

    .particle {
        position: absolute;
        background: radial-gradient(circle, rgba(0,255,255,0.8) 0%, rgba(0,255,255,0.03) 70%);
        border-radius: 50%;
        opacity: 0.7;
        animation: floatParticles linear infinite;
    }

    @keyframes floatParticles {
        from { transform: translateY(0) scale(1); opacity: 0.9; }
        to { transform: translateY(-180px) scale(1.4); opacity: 0.2; }
    }

    /* =======================================================
       🌊 NAVBAR GLASS PANEL
    ======================================================= */
    .navbar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 70px;
        background: linear-gradient(135deg, rgba(0,40,60,0.7), rgba(0,80,100,0.4));
        backdrop-filter: blur(18px);
        border-bottom: 1px solid rgba(0,255,255,0.25);
        box-shadow: 0 0 35px rgba(0,200,255,0.3);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 40px;
        z-index: 999;
        animation: fadeSlide 0.7s ease;
    }

    @keyframes fadeSlide {
        0% {opacity:0; transform:translateY(-15px);}
        100% {opacity:1; transform:translateY(0);}
    }

    .nav-left {
        display: flex;
        align-items: center;
        gap: 22px;
    }

    .nav-item {
        font-size: 17px;
        font-weight: 600;
        color: #bde0fe;
        text-decoration: none;
        transition: all 0.3s ease;
        letter-spacing: 0.3px;
        position: relative;
    }

    .nav-item:hover {
        color: #00e0ff;
        text-shadow: 0 0 18px #00e0ff;
        transform: translateY(-1px);
    }

    .nav-item-active {
        color: #00ffff;
        text-shadow: 0 0 20px #00e0ff;
        border-bottom: 2px solid #00e0ff;
        padding-bottom: 3px;
    }

    /* =======================================================
       👤 PROFILE CARD (RIGHT SIDE)
    ======================================================= */
    .nav-profile {
        display: flex;
        align-items: center;
        gap: 14px;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(0,255,255,0.25);
        border-radius: 50px;
        padding: 8px 16px 8px 18px;
        box-shadow: 0 0 18px rgba(0,200,255,0.25);
        transition: all 0.4s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }

    .nav-profile::before {
        content: "";
        position: absolute;
        width: 90%;
        height: 260%;
        background: linear-gradient(120deg, rgba(0,255,255,0.35), rgba(0,255,255,0));
        top: -75%;
        left: -120%;
        transform: rotate(25deg);
        transition: 0.5s;
    }

    .nav-profile:hover::before {
        left: 130%;
    }

    .nav-profile:hover {
        background: rgba(0,100,150,0.3);
        transform: scale(1.05);
        box-shadow: 0 0 40px rgba(0,240,255,0.4);
    }

    .profile-name {
        font-weight: 700;
        color: #00e0ff;
        font-size: 16px;
        margin-bottom: -3px;
        text-shadow: 0 0 12px #00e0ff;
    }

    .profile-role {
        font-size: 13px;
        color: #8ecae6;
    }

    .profile-buttons {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 18px;
        color: #bde0fe;
    }

    .profile-buttons span {
        transition: all 0.25s ease;
    }

    .profile-buttons span:hover {
        color: #00e0ff;
        text-shadow: 0 0 10px #00e0ff;
        transform: scale(1.2);
    }

    /* 🌟 Soft Glow Pulse */
    .nav-glow {
        animation: glowPulse 3s ease-in-out infinite;
    }

    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 20px rgba(0,200,255,0.25); }
        50% { box-shadow: 0 0 50px rgba(0,255,255,0.45); }
    }
    </style>
    """, unsafe_allow_html=True)

    # 🌌 Full Background Particle Layer (20 dots)
    particle_html = '<div class="global-bg">'
    for i in range(20):
        left = f"{(i * 5 + 3) % 100}%"
        top = f"{(i * 9 + 15) % 100}%"
        size = 6 + (i % 4) * 3
        duration = 5 + (i % 5) * 3
        particle_html += f'<div class="particle" style="left:{left};top:{top};width:{size}px;height:{size}px;animation-duration:{duration}s;"></div>'
    particle_html += '</div>'

    # 🌊 Navbar Layout
    nav_html = '<div class="navbar nav-glow"><div class="nav-left">'
    for icon, page in nav_items:
        active_class = "nav-item-active" if page == current_page else "nav-item"
        page_path = page.replace(" ", "_")
        nav_html += f'<a class="{active_class}" href="/{page_path}">{icon}</a>'
    nav_html += '</div>'

    # 👤 Profile Card
    nav_html += f"""
    <div class="nav-profile">
        <div>
            <div class="profile-name">{user_name}</div>
            <div class="profile-role">{user_role}</div>
        </div>
        <div class="profile-buttons">
            <span title="Settings">⚙️</span>
            <span title="Logout">🚪</span>
        </div>
    </div>
    </div>
    """

    # Combine all layers
    st.markdown(particle_html + nav_html, unsafe_allow_html=True)
    st.markdown("<br><br><br>", unsafe_allow_html=True)
