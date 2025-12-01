# ================================================================
# utils/analytics_modes.py — CyberGlass Aqua v15.5 Stable
# Unified Visual Modes: Dataset Analytics • Regional Radar • Network Graph
# ================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import networkx as nx
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------
# 1️⃣ DATA ANALYTICS MODE — dataset-based exploration
# ---------------------------------------------------------------
def data_analytics_mode(df, crime_cols, socio_cols, auto_region=None, auto_year=None):
    st.subheader("📊 Data Analytics Explorer")

    analysis_mode = st.selectbox(
        "Select Analysis Type",
        ["Correlation Heatmap", "Scatter Explorer", "Yearly Trend", "3D Relationship Matrix", "Smart Summary"],
        index=0
    )

    crime_var = st.selectbox("Select Crime Metric", crime_cols, index=0)
    x_var = st.selectbox("Select Socio-Economic Factor (X)", socio_cols, index=0)
    y_var = st.selectbox("Optional Y Factor (for 3D)", ["None"] + socio_cols, index=0)

    # ----------------------------------------------------------------
    # Correlation Heatmap
    # ----------------------------------------------------------------
    if analysis_mode == "Correlation Heatmap":
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr().round(2)
        fig = ff.create_annotated_heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            annotation_text=corr.values,
            colorscale="Tealgrn",
            showscale=True
        )
        fig.update_layout(template="plotly_dark", height=650, title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"""
            🧠 **Insight:**  
            The heatmap shows how features relate numerically.  
            - Darker greens = stronger positive correlation  
            - Darker blues = stronger negative correlation  
            A high |r| (>0.7) suggests the variable could strongly influence crime rates.
            """
        )

    # ----------------------------------------------------------------
    # Scatter Explorer
    # ----------------------------------------------------------------
    elif analysis_mode == "Scatter Explorer":
        fig = px.scatter(
            df, x=x_var, y=crime_var,
            color="state" if "state" in df.columns else None,
            trendline="ols", template="plotly_dark", opacity=0.7
        )
        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="#00b4d8")))
        st.plotly_chart(fig, use_container_width=True)

        # Linear trend interpretation
        X = df[[x_var]].dropna()
        Y = df[crime_var].loc[X.index]
        if len(X) > 10:
            slope = LinearRegression().fit(X, Y).coef_[0]
            relation = "positive" if slope > 0 else "negative"
            st.success(
                f"📈 **Trendline Interpretation:** {crime_var.title()} has a {relation} relationship with {x_var.title()} (slope = {slope:.3f})."
            )
        else:
            st.warning("Not enough data points for reliable regression.")

    # ----------------------------------------------------------------
    # Yearly Trend
    # ----------------------------------------------------------------
    elif analysis_mode == "Yearly Trend":
        if "year" in df.columns:
            yearly = df.groupby("year")[crime_var].mean().reset_index()
            fig = px.line(
                yearly, x="year", y=crime_var,
                markers=True, color_discrete_sequence=["#00e0ff"],
                template="plotly_dark"
            )
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(fig, use_container_width=True)
            slope = yearly[crime_var].diff().mean()
            direction = "increasing" if slope > 0 else "decreasing"
            st.info(f"📅 **Trend Insight:** On average, {crime_var.title()} is {direction} over time.")
        else:
            st.warning("⚠️ Dataset has no 'year' column for trend visualization.")

    # ----------------------------------------------------------------
    # 3D Relationship Matrix
    # ----------------------------------------------------------------
    elif analysis_mode == "3D Relationship Matrix":
        if y_var != "None" and y_var in df.columns:
            fig = px.scatter_3d(
                df, x=x_var, y=y_var, z=crime_var,
                color=crime_var, color_continuous_scale="Teal", template="plotly_dark"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🌐 This 3D plot reveals multivariate relationships between factors and crime intensity.")
        else:
            st.info("Please select a valid Y variable for 3D visualization.")

    # ----------------------------------------------------------------
    # Smart Summary
    # ----------------------------------------------------------------
    elif analysis_mode == "Smart Summary":
        desc = df[crime_var].describe().to_frame()
        st.dataframe(desc.style.highlight_max(color="#00b4d8"))
        fig = px.histogram(df, x=crime_var, nbins=25, color_discrete_sequence=["#00e0ff"], template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"📦 Statistical summary and distribution of {crime_var.title()}.")


# ---------------------------------------------------------------
# 2️⃣ REGIONAL INTELLIGENCE MODE — radar & deviation analysis
# ---------------------------------------------------------------
def regional_intelligence_mode(df, socio_cols, auto_region=None):
    st.subheader("🌍 Regional Intelligence Studio")

    geo_type = st.radio("Geographic Level", ["State", "District"], horizontal=True)
    region_col = "district" if geo_type == "District" and "district" in df.columns else "state_name"
    regions = sorted(df[region_col].dropna().unique())
    default_idx = 0
    if auto_region and auto_region in regions:
        default_idx = regions.index(auto_region)

    region = st.selectbox(f"Select {geo_type}", regions, index=default_idx)

    region_df = df[df[region_col] == region]
    if region_df.empty:
        st.warning("⚠️ No data for this region.")
        return

    avg_nat = df[socio_cols].mean()
    avg_reg = region_df[socio_cols].mean()
    comparison = ((avg_reg - avg_nat) / avg_nat * 100).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Radar chart
    top_feats = comparison.abs().sort_values(ascending=False).head(6).index.tolist()
    radar_df = pd.DataFrame({
        "Variable": top_feats,
        "Region": avg_reg[top_feats].values,
        "National": avg_nat[top_feats].values
    })

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=radar_df["Region"], theta=radar_df["Variable"],
                                        fill='toself', name=region, line=dict(color="#00e0ff")))
    fig_radar.add_trace(go.Scatterpolar(r=radar_df["National"], theta=radar_df["Variable"],
                                        fill='toself', name="National Avg", line=dict(color="#ff5a5a")))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), template="plotly_dark", height=600)
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption("💠 Radar chart comparing key regional metrics vs. national averages.")

    # Bar deviation
    rank_df = comparison[top_feats].sort_values(ascending=False).reset_index()
    rank_df.columns = ["Indicator", "Deviation (%)"]
    rank_df["Category"] = np.where(rank_df["Deviation (%)"] > 0, "Above Avg", "Below Avg")
    fig_bar = px.bar(
        rank_df, x="Indicator", y="Deviation (%)", color="Category",
        color_discrete_map={"Above Avg": "#00e0ff", "Below Avg": "#ff5a5a"},
        text="Deviation (%)", template="plotly_dark"
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("📊 Bar chart showing deviation of indicators from national averages.")

    pos = comparison[comparison > 0].head(2).index.tolist()
    neg = comparison[comparison < 0].tail(2).index.tolist()
    st.markdown(f"""
    🧠 **AI-style Summary for {region}**
    - Strengths: {', '.join(pos) if pos else '—'}  
    - Weak Areas: {', '.join(neg) if neg else '—'}  
    - Suggestion: Focus investments on underperforming indicators to balance development.
    """)


# ---------------------------------------------------------------
# 3️⃣ CORRELATION NETWORK MODE — feature graph
# ---------------------------------------------------------------
def correlation_network_mode(df, crime_cols, socio_cols):
    st.subheader("🕸️ Correlation Network Visualization")

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    threshold = st.slider("Correlation Threshold", 0.3, 1.0, 0.6, 0.05)
    edges = [(i, j, corr.loc[i, j]) for i in corr.columns for j in corr.columns
             if i != j and abs(corr.loc[i, j]) >= threshold]

    if not edges:
        st.warning("No correlations above the threshold.")
        return

    G = nx.Graph()
    for i, j, w in edges:
        G.add_edge(i, j, weight=abs(w))

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y, node_x, node_y, node_text = [], [], [], [], []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#00b4d8"),
                            hoverinfo="none", mode="lines")
    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", text=node_text,
                            textposition="top center", hoverinfo="text",
                            marker=dict(size=12, color="#00e0ff"))

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(template="plotly_dark",
                         title="Feature Correlation Network",
                         showlegend=False, hovermode="closest")
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔗 Nodes represent dataset variables. Edges connect pairs with |correlation| ≥ threshold.")
