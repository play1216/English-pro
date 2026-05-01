import html

import streamlit as st


def apply_app_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(29, 78, 216, 0.08), transparent 28%),
                radial-gradient(circle at top right, rgba(16, 185, 129, 0.08), transparent 24%),
                linear-gradient(180deg, #f8fbff 0%, #f3f7fb 100%);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #162033 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }

        section[data-testid="stSidebar"] * {
            color: #e5eefb;
        }

        section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
            color: #ffffff;
        }

        .app-hero {
            padding: 1.35rem 1.5rem;
            border-radius: 18px;
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.96) 0%, rgba(30, 41, 59, 0.9) 52%, rgba(18, 92, 163, 0.82) 100%);
            color: #f8fafc;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.14);
            border: 1px solid rgba(148, 163, 184, 0.18);
            margin-bottom: 1rem;
        }

        .app-hero h1 {
            font-size: 2rem;
            line-height: 1.15;
            margin: 0 0 0.4rem 0;
            letter-spacing: 0;
        }

        .app-hero p {
            margin: 0;
            color: rgba(226, 232, 240, 0.92);
            font-size: 0.98rem;
        }

        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.9rem;
        }

        .hero-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.36rem 0.72rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.12);
            font-size: 0.9rem;
            color: #f8fafc;
        }

        .panel-title {
            font-size: 0.92rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.45rem;
        }

        .panel-tip {
            font-size: 0.9rem;
            color: #475569;
            margin-bottom: 0.75rem;
        }

        .score-card {
            padding: 1.2rem 1rem;
            border-radius: 18px;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 18px 34px rgba(15, 23, 42, 0.08);
            text-align: center;
            margin-bottom: 0.85rem;
        }

        .score-rank {
            font-size: 1rem;
            font-weight: 700;
            color: #334155;
            margin-bottom: 0.25rem;
        }

        .score-value {
            font-size: 3.3rem;
            line-height: 1;
            font-weight: 800;
            color: #0f766e;
        }

        .score-max {
            font-size: 1.15rem;
            color: #64748b;
            margin-left: 0.35rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 14px;
            padding: 0.7rem 0.8rem;
        }

        div[data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px dashed rgba(59, 130, 246, 0.35);
            border-radius: 16px;
            padding: 0.35rem;
        }

        .stTextArea textarea,
        .stTextInput input {
            border-radius: 14px !important;
            border: 1px solid rgba(148, 163, 184, 0.35) !important;
            background: rgba(255, 255, 255, 0.92) !important;
        }

        .stButton > button {
            border-radius: 14px !important;
            border: 0 !important;
            background: linear-gradient(135deg, #1d4ed8 0%, #0f766e 100%) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            box-shadow: 0 10px 22px rgba(29, 78, 216, 0.18);
        }

        .stExpander {
            border: 1px solid rgba(148, 163, 184, 0.18) !important;
            border-radius: 14px !important;
            background: rgba(255, 255, 255, 0.72);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header(title, subtitle, current_user, essay_type_label):
    safe_title = html.escape(title)
    safe_subtitle = html.escape(subtitle)
    safe_user = html.escape(current_user or "未登录")
    safe_type = html.escape(essay_type_label)

    st.markdown(
        f"""
        <div class="app-hero">
            <h1>{safe_title}</h1>
            <p>{safe_subtitle}</p>
            <div class="hero-meta">
                <span class="hero-chip">当前用户：{safe_user}</span>
                <span class="hero-chip">作文类型：{safe_type}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel_title(title, tip=""):
    safe_title = html.escape(title)
    safe_tip = html.escape(tip)
    st.markdown(f'<div class="panel-title">{safe_title}</div>', unsafe_allow_html=True)
    if safe_tip:
        st.markdown(f'<div class="panel-tip">{safe_tip}</div>', unsafe_allow_html=True)


def render_score_card(rank, total, max_score):
    safe_rank = html.escape(str(rank))
    safe_total = html.escape(str(total))
    safe_max_score = html.escape(str(max_score))
    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-rank">{safe_rank}</div>
            <div>
                <span class="score-value">{safe_total}</span>
                <span class="score-max">/ {safe_max_score}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
