"""BullsAI 공통 UI 테마 — Pretendard 기반, 이모지 없는 차분한 B2B/공공 톤.

두 Streamlit 페이지(메인, 라벨링)에서 `apply_theme()`를 호출해 동일한
글꼴·색상·컴포넌트 스타일을 적용한다. 색상은 zinc 중립 + 단일 딥블루 accent.
"""
from __future__ import annotations

import streamlit as st

# 디자인 토큰
ACCENT = "#1d4ed8"          # 딥블루 (신뢰형 B2B/공공)
ACCENT_HOVER = "#1e40af"
INK = "#18181b"             # 본문 텍스트 (zinc-900)
INK_MUTED = "#52525b"       # 보조 텍스트 (zinc-600)
INK_FAINT = "#a1a1aa"       # 뮤트 (zinc-400)
LINE = "rgba(24,24,27,0.08)"   # hairline 경계
SURFACE = "#ffffff"
CANVAS = "#fafafa"          # zinc-50

_CSS = f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

:root {{
    --ba-accent: {ACCENT};
    --ba-accent-hover: {ACCENT_HOVER};
    --ba-ink: {INK};
    --ba-ink-muted: {INK_MUTED};
    --ba-line: {LINE};
}}

html, body, [class*="css"], .stApp,
button, input, textarea, select {{
    font-family: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont,
        "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}

.stApp {{ background: {CANVAS}; }}

/* 헤딩: bold 금지, semibold + tracking-tight */
h1, h2, h3, h4 {{
    color: {INK};
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}}
h1 {{ font-size: 1.9rem !important; }}
h2 {{ font-size: 1.35rem !important; }}
h3 {{ font-size: 1.1rem !important; }}

p, span, label, li, .stMarkdown {{ color: {INK}; }}

/* 본문/캡션 대비 계층 */
.stCaption, [data-testid="stCaptionContainer"], small {{
    color: {INK_MUTED} !important;
    font-size: 0.82rem;
}}

/* 사이드바 */
[data-testid="stSidebar"] {{
    background: {SURFACE};
    border-right: 1px solid {LINE};
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] .stHeadingContainer {{ font-size: 1rem !important; }}

/* 탭 — 밑줄형, accent 강조 */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.25rem;
    border-bottom: 1px solid {LINE};
}}
.stTabs [data-baseweb="tab"] {{
    font-weight: 500;
    color: {INK_MUTED};
    padding: 0.5rem 0.9rem;
}}
.stTabs [aria-selected="true"] {{
    color: {ACCENT} !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {ACCENT}; }}

/* 기본(primary) 버튼 — 단일 solid, accent */
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    color: #fff;
    font-weight: 500;
    border-radius: 8px;
    box-shadow: none;
    transition: background 160ms ease;
}}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {{
    background: {ACCENT_HOVER};
    border-color: {ACCENT_HOVER};
    color: #fff;
}}

/* 보조 버튼 — outline/ghost */
.stButton > button:not([kind="primary"]),
.stDownloadButton > button:not([kind="primary"]) {{
    background: {SURFACE};
    border: 1px solid {LINE};
    color: {INK};
    font-weight: 500;
    border-radius: 8px;
    box-shadow: none;
}}
.stButton > button:not([kind="primary"]):hover,
.stDownloadButton > button:not([kind="primary"]):hover {{
    border-color: {ACCENT};
    color: {ACCENT};
}}

/* 입력 요소 */
[data-baseweb="input"] input,
[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {{
    border-radius: 8px !important;
}}

/* metric 카드 */
[data-testid="stMetric"] {{
    background: {SURFACE};
    border: 1px solid {LINE};
    border-radius: 10px;
    padding: 0.9rem 1rem;
}}
[data-testid="stMetricLabel"] {{ color: {INK_MUTED} !important; }}
[data-testid="stMetricValue"] {{
    color: {INK} !important;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}}

/* expander — hairline, 그림자 제거 */
[data-testid="stExpander"] {{
    border: 1px solid {LINE};
    border-radius: 10px;
    box-shadow: none;
    background: {SURFACE};
}}

/* 알림 박스 — 톤 다운 */
[data-testid="stAlert"] {{
    border-radius: 10px;
    border: 1px solid {LINE};
}}

/* 표 */
[data-testid="stDataFrame"] {{ border-radius: 10px; }}

/* 상단 여백 정리 */
.block-container {{ padding-top: 2.2rem; }}

/* 브랜드 워드마크 */
.ba-brand {{
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
}}
.ba-brand .ba-name {{
    font-size: 1.9rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    color: {INK};
}}
.ba-brand .ba-name b {{ color: {ACCENT}; font-weight: 600; }}
.ba-brand .ba-tag {{
    font-size: 0.9rem;
    color: {INK_MUTED};
    font-weight: 500;
}}
</style>
"""


def apply_theme() -> None:
    """전역 CSS와 글꼴을 주입한다. 각 페이지 상단에서 한 번 호출."""
    st.markdown(_CSS, unsafe_allow_html=True)


def brand_header(tagline: str = "") -> None:
    """BullsAI 워드마크 헤더를 렌더링한다 (이모지 없음)."""
    tag = f'<span class="ba-tag">{tagline}</span>' if tagline else ""
    st.markdown(
        f'<div class="ba-brand"><span class="ba-name">Bulls<b>AI</b></span>{tag}</div>',
        unsafe_allow_html=True,
    )
