"""BullsAI theme — 앱 로고 톤: 차콜 · 스틸 · 레드 · 파도 블루. 그라데이션 없이 단색으로.

로고 이미지는 app/static/bullsai_logo.png 원본을 그대로 쓴다 (자르기·색 보정·합성 금지).
"""
from __future__ import annotations

import base64
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st

LOGO = Path(__file__).resolve().parent / "static" / "bullsai_logo.png"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Black+Ops+One&family=IBM+Plex+Mono:wght@500;700&family=Noto+Sans+KR:wght@400;500;700;800;900&display=swap');
:root{
  --bg:#161618; --panel:#1f1f22; --panel2:#26262a; --line:#323237; --line2:#44444b;
  --steel:#d0cdc5; --steel2:#8f8a82; --ink:#ecebe8; --muted:#8f8a82;
  --red:#b3261e; --red2:#cf3a30; --blue:#304a5a; --blue2:#4d6f84;
  --ok:#4f9d6c; --warn:#c8963a;
}
html,body,.stApp,button,input,textarea,select{font-family:"Noto Sans KR",system-ui,sans-serif !important}
.stApp{color:var(--ink);background:var(--bg)}
header[data-testid="stHeader"]{background:transparent !important;pointer-events:none}
header[data-testid="stHeader"] [data-testid="stToolbar"]{pointer-events:none}
header[data-testid="stHeader"] button,header[data-testid="stHeader"] a{pointer-events:auto}
[data-testid="stDecoration"],[data-testid="stToolbarActions"],[data-testid="stMainMenu"],[data-testid="stAppDeployButton"],#MainMenu,footer{display:none !important}
[data-testid="stExpandSidebarButton"]{background:var(--panel2) !important;border:1px solid var(--line2) !important;border-radius:6px !important;color:var(--steel) !important}
.block-container{max-width:1200px;padding-top:1.4rem;padding-bottom:5rem}
h1,h2,h3,h4,p,label,li,.stMarkdown,[data-testid="stMetricValue"]{color:var(--ink) !important}
h1,h2,h3,[data-testid="stHeadingWithActionElements"] *{font-weight:800 !important;letter-spacing:0}
h2{font-size:1.6rem !important} h3{font-size:1.3rem !important}
[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label{font-weight:700 !important;color:var(--steel) !important;font-size:.9rem !important}
.stCaption,[data-testid="stCaptionContainer"],small{color:var(--muted) !important}
a{color:var(--steel) !important} hr{border-color:var(--line) !important}

/* sidebar */
[data-testid="stSidebar"]{background:#121214;border-right:1px solid var(--line)}
[data-testid="stSidebar"] > div:first-child{padding-top:.6rem}
[data-testid="stSidebarUserContent"]{padding-top:.4rem}
[data-testid="stSidebarNav"]{display:none}
[data-testid="stSidebar"] [data-testid="stPageLink"] a{border-radius:6px;padding:.35rem .6rem}
[data-testid="stSidebar"] [data-testid="stPageLink"] a p{font-weight:800 !important;font-size:1.05rem !important}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover{background:var(--panel2)}
.ba-sec{color:var(--red2);font:700 .68rem "IBM Plex Mono",monospace;letter-spacing:.18em;margin:1.3rem 0 .45rem;padding-bottom:.35rem;border-bottom:1px solid var(--line)}

/* brand */
.ba-brand{padding:0 0 .9rem;border-bottom:1px solid var(--line);margin-bottom:.5rem}
.ba-brand img{display:block;width:100%;height:auto;border-radius:10px;border:1px solid var(--line)}
.ba-name{font:400 1.55rem "Black Ops One",monospace;color:var(--steel);letter-spacing:.03em}
.ba-name b{color:var(--red2);font-weight:inherit}

/* logo banner */
.ba-banner{display:flex;justify-content:center;align-items:center;background:#1a1a1c;border:1px solid var(--line);border-radius:12px;padding:.6rem;margin-bottom:1.4rem}
.ba-banner img{display:block;width:100%;max-width:520px;height:auto;border-radius:8px}
.ba-title{font:800 1.8rem "Noto Sans KR",sans-serif;color:var(--ink);margin:.2rem 0 1.2rem;padding-left:.8rem;border-left:4px solid var(--red)}

/* tabs */
.stTabs [role="tablist"]{gap:.4rem;border-bottom:1px solid var(--line);padding-bottom:.6rem;flex-wrap:wrap}
.stTabs [role="tab"]{color:var(--muted) !important;background:var(--panel) !important;border:1px solid var(--line) !important;border-radius:6px !important;padding:.5rem 1.1rem !important;height:auto !important}
.stTabs [role="tab"] p{color:inherit !important;font-weight:800 !important;font-size:1rem !important}
.stTabs [role="tab"]:hover{color:var(--ink) !important;border-color:var(--line2) !important}
.stTabs [role="tab"][aria-selected="true"]{color:#fff !important;background:var(--red) !important;border-color:var(--red) !important}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none !important}
[data-testid="stTabContent"]{padding-top:1.2rem}
.ba-step{display:flex;align-items:center;gap:.55rem;color:var(--steel2);font:700 .7rem "IBM Plex Mono",monospace;letter-spacing:.16em;margin:0 0 .8rem}
.ba-step b{color:#fff;background:var(--red);border-radius:4px;padding:.1rem .45rem}

/* buttons: flat */
.stButton > button,.stDownloadButton > button{min-height:2.7rem;border-radius:6px !important;font-weight:800 !important;font-size:1rem !important;box-shadow:none !important;transition:background .12s,border-color .12s}
.stButton > button p,.stDownloadButton > button p{font-weight:800 !important}
.stButton > button[kind="primary"],.stDownloadButton > button[kind="primary"]{color:#fff !important;background:var(--red) !important;border:1px solid var(--red) !important}
.stButton > button[kind="primary"]:hover,.stDownloadButton > button[kind="primary"]:hover{background:var(--red2) !important;border-color:var(--red2) !important}
.stButton > button:not([kind="primary"]),.stDownloadButton > button:not([kind="primary"]){color:var(--ink) !important;background:var(--panel2) !important;border:1px solid var(--line2) !important}
.stButton > button:not([kind="primary"]):hover,.stDownloadButton > button:not([kind="primary"]):hover{border-color:var(--steel2) !important}

/* inputs */
[data-baseweb="input"],[data-baseweb="select"] > div,[data-baseweb="textarea"],[data-testid="stNumberInputContainer"],[data-testid="stTextInputRootElement"]{background:#131315 !important;border:1px solid var(--line2) !important;border-radius:6px !important}
[data-baseweb="base-input"],[data-testid="stTextInputRootElement"] > div{background:transparent !important;border:0 !important}
[data-baseweb="input"]:focus-within,[data-baseweb="select"] > div:focus-within,[data-testid="stTextInputRootElement"]:focus-within{border-color:var(--steel2) !important}
input,textarea,[data-baseweb="select"] span,[data-baseweb="select"] div{color:var(--ink) !important;-webkit-text-fill-color:var(--ink) !important}
[data-testid="stNumberInput"] button{background:var(--panel2) !important;color:var(--steel) !important;border-color:var(--line2) !important}
[data-baseweb="popover"] ul,[data-baseweb="menu"]{background:var(--panel) !important;border:1px solid var(--line2) !important}
[data-baseweb="popover"] li{color:var(--ink) !important} [data-baseweb="popover"] li:hover,[aria-selected="true"][role="option"]{background:var(--panel2) !important}
[data-testid="stFileUploaderDropzone"]{background:#131315 !important;border:1px dashed var(--line2) !important;border-radius:8px !important;padding:1.1rem}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--steel2) !important}
[data-testid="stFileUploaderDropzone"] span,[data-testid="stFileUploaderDropzone"] small{color:var(--muted) !important}
[data-testid="stFileUploaderDropzone"] button{background:var(--panel2) !important;color:var(--ink) !important;border:1px solid var(--line2) !important}

/* cards, metrics, alerts */
[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--red);border-radius:8px;padding:.9rem 1rem}
[data-testid="stMetricLabel"] p{color:var(--muted) !important;font-weight:700 !important;font-size:.82rem !important}
[data-testid="stMetricValue"]{font:700 clamp(1.3rem,2vw,1.8rem) "IBM Plex Mono",monospace !important;color:#fff !important}
[data-testid="stExpander"]{background:var(--panel);border:1px solid var(--line) !important;border-radius:8px}
[data-testid="stExpander"] summary p{color:var(--ink) !important;font-weight:800 !important;font-size:1.02rem !important}
[data-testid="stExpander"] summary:hover p{color:var(--red2) !important}
[data-testid="stAlert"],[data-testid="stAlert"] > div,[data-testid="stAlertContainer"]{border-radius:8px;background:#1b2429 !important;color:var(--ink) !important}
[data-testid="stAlert"]{border:1px solid #304a5a;border-left:3px solid #4d6f84}
[data-testid="stImage"] img{border-radius:8px;border:1px solid var(--line)}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:8px;overflow:hidden}

@media(max-width:820px){
  .block-container{padding:3.4rem .9rem 3rem}
  .ba-banner img{max-width:100%}
  .stTabs [role="tab"]{padding:.45rem .7rem !important}
}
</style>
"""


@lru_cache(maxsize=1)
def logo_data_uri() -> str | None:
    if not LOGO.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(LOGO.read_bytes()).decode()


def page_icon():
    return str(LOGO) if LOGO.exists() else "🎯"


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def brand_header() -> None:
    uri = logo_data_uri()
    # 로고 안에 이름이 들어 있으므로 로고만 크게. 로고 파일이 없을 때만 글자 이름.
    inner = f'<img src="{uri}" alt="BullsAI">' if uri else '<span class="ba-name">BULLS<b>AI</b></span>'
    st.markdown(f'<div class="ba-brand">{inner}</div>', unsafe_allow_html=True)


def sidebar_nav() -> None:
    try:
        st.page_link("streamlit_app.py", label="사격 분석")
        st.page_link("pages/1_라벨링.py", label="탄공 라벨링")
    except Exception:  # noqa: BLE001 — page opened on its own (e.g. in tests): no menu
        pass


def section(label: str) -> None:
    st.markdown(f'<div class="ba-sec">{escape(label)}</div>', unsafe_allow_html=True)


def step(num: str, label: str) -> None:
    st.markdown(f'<div class="ba-step"><b>{escape(num)}</b>{escape(label)}</div>', unsafe_allow_html=True)


def banner() -> None:
    """Main-page logo banner (the logo already carries the app name, so no extra copy)."""
    uri = logo_data_uri()
    if uri:
        st.markdown(f'<div class="ba-banner"><img src="{uri}" alt="BullsAI"></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ba-title">BULLSAI</div>', unsafe_allow_html=True)


def page_title(title: str) -> None:
    st.markdown(f'<div class="ba-title">{escape(title)}</div>', unsafe_allow_html=True)
