"""BullsAI theme — SSAT(해경 특공대) 마크 톤: 다크 네이비 카모 · 스틸 실버 · 딥 오션 블루 · 삼지창 골드.

마크 이미지는 app/static/brand/ssat_emblem.png 원본을 그대로 쓴다 (자르기·색 보정·합성 금지).
파일이 없으면(공개 저장소 클론 등) 텍스트 로고로 대체한다.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st

EMBLEM = Path(__file__).resolve().parent / "static" / "brand" / "ssat_emblem.png"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800&family=JetBrains+Mono:wght@500;700&family=Noto+Sans+KR:wght@400;500;700;900&display=swap');
:root{
  --bg:#0b0e14; --bg2:#10151e; --panel:#141a24; --panel2:#19212d;
  --steel:#c9ced6; --steel2:#8b95a3; --line:#263041; --line2:#344155;
  --blue:#2f6fa8; --blue2:#4d8fcc; --deep:#173d62; --gold:#d6a936; --gold2:#f0c75a;
  --ink:#e8ecf1; --muted:#8e98a7; --ok:#3fb27f; --warn:#e0a13a; --bad:#e0555a;
}
html,body,.stApp,button,input,textarea,select{font-family:"Noto Sans KR",system-ui,sans-serif !important}
.stApp{color:var(--ink);background-color:var(--bg);
  background-image:
    radial-gradient(ellipse 70% 55% at 50% -10%,#1d3a5c55,transparent 70%),
    radial-gradient(ellipse 60% 50% at 100% 100%,#15283f44,transparent 70%),
    linear-gradient(#ffffff05 1px,transparent 1px),
    linear-gradient(90deg,#ffffff05 1px,transparent 1px),
    linear-gradient(#ffffff03 1px,transparent 1px),
    linear-gradient(90deg,#ffffff03 1px,transparent 1px);
  background-size:100% 100%,100% 100%,96px 96px,96px 96px,24px 24px,24px 24px;
  background-attachment:fixed}
header[data-testid="stHeader"]{background:transparent !important;height:0}
[data-testid="stToolbar"],[data-testid="stDecoration"],#MainMenu,footer{display:none !important}
.block-container{max-width:1240px;padding-top:1.6rem;padding-bottom:5rem}
h1,h2,h3,h4,p,label,li,.stMarkdown,[data-testid="stMetricValue"]{color:var(--ink) !important}
h2,h3{letter-spacing:-.02em;font-weight:800 !important}
.stCaption,[data-testid="stCaptionContainer"],small{color:var(--muted) !important}
a{color:var(--blue2) !important} hr{border-color:var(--line) !important}

/* sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0e131b,#0a0d13);border-right:1px solid var(--line)}
[data-testid="stSidebar"] > div:first-child{padding-top:1rem}
[data-testid="stSidebarNav"]{display:none}
[data-testid="stSidebar"] [data-testid="stPageLink"] a{border-radius:6px;border:1px solid transparent;padding:.35rem .6rem}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover{border-color:var(--line2);background:#ffffff06}
.ba-sec{color:var(--gold);font:700 .68rem "JetBrains Mono",monospace;letter-spacing:.18em;margin:1.4rem 0 .5rem;display:flex;align-items:center;gap:.5rem}
.ba-sec:after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line2),transparent)}

/* brand */
.ba-brand{display:flex;align-items:center;gap:.75rem;padding:.4rem 0 .9rem;border-bottom:1px solid var(--line);margin-bottom:.4rem}
.ba-brand img{width:64px;height:auto;border-radius:6px;box-shadow:0 0 0 1px #3a4658,0 6px 18px #000a}
.ba-mark{width:44px;height:44px;border-radius:50%;flex:none;border:2px solid var(--steel2);background:radial-gradient(circle,var(--gold) 0 14%,#0d1117 15% 34%,var(--blue) 35% 50%,#0d1117 51% 68%,var(--steel2) 69%)}
.ba-name{font:800 1.45rem "Cinzel",serif;letter-spacing:.08em;background:linear-gradient(180deg,#f3f5f8,#9aa3b0);-webkit-background-clip:text;background-clip:text;color:transparent}
.ba-name b{background:linear-gradient(180deg,var(--gold2),#b58a24);-webkit-background-clip:text;background-clip:text;color:transparent;font-weight:inherit}
.ba-tag{display:block;color:var(--muted);font:600 .62rem "JetBrains Mono",monospace;letter-spacing:.16em;margin-top:.1rem}

/* hero */
.ba-hero{position:relative;overflow:hidden;display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,.85fr);gap:2rem;align-items:center;
  border:1px solid var(--line2);border-radius:14px;padding:clamp(1.4rem,3vw,2.4rem);margin-bottom:1.6rem;
  background:linear-gradient(120deg,#131a25 0%,#0f141d 55%,#0c1a2b 100%);
  box-shadow:0 24px 60px #000b,inset 0 1px 0 #ffffff0d}
.ba-hero:before{content:"";position:absolute;inset:0;pointer-events:none;
  background:repeating-linear-gradient(90deg,#ffffff03 0 1px,transparent 1px 64px),repeating-linear-gradient(0deg,#ffffff03 0 1px,transparent 1px 64px)}
.ba-hero:after{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:linear-gradient(180deg,var(--gold),var(--blue))}
.ba-eyebrow{display:inline-flex;align-items:center;gap:.5rem;color:var(--gold);font:700 .72rem "JetBrains Mono",monospace;letter-spacing:.2em}
.ba-eyebrow:before{content:"";width:8px;height:8px;border-radius:50%;background:var(--gold);box-shadow:0 0 10px var(--gold)}
.ba-hero h1{position:relative;z-index:1;margin:.9rem 0 .6rem;font:900 clamp(2rem,4.2vw,3.3rem)/1.12 "Noto Sans KR",sans-serif !important;letter-spacing:-.035em;
  background:linear-gradient(180deg,#ffffff,#b8c1cd);-webkit-background-clip:text;background-clip:text;color:transparent !important}
.ba-hero > div:first-child{min-width:0}
.ba-hero p{position:relative;z-index:1;max-width:100%;color:#aeb7c4 !important;line-height:1.8;margin:0}
.ba-chips{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1.3rem;position:relative;z-index:1}
.ba-chip{font:600 .72rem "JetBrains Mono",monospace;letter-spacing:.06em;color:var(--steel);border:1px solid var(--line2);background:#0c1119cc;border-radius:999px;padding:.35rem .75rem}
.ba-chip b{color:var(--gold2);font-weight:700}
.ba-emblem{position:relative;z-index:1;justify-self:end;width:100%;max-width:440px;margin:0}
.ba-emblem img{display:block;width:100%;height:auto;border-radius:10px;
  box-shadow:0 0 0 1px #3d4a5e,0 0 0 6px #0b0f16,0 0 0 7px #2a3547,0 26px 60px #000c,0 0 80px #2f6fa822}
.ba-emblem figcaption{margin-top:.8rem;text-align:center;color:var(--muted);font:600 .64rem "JetBrains Mono",monospace;letter-spacing:.22em}

/* tabs */
.stTabs [role="tablist"]{gap:.45rem;border-bottom:1px solid var(--line);padding-bottom:.7rem;flex-wrap:wrap}
.stTabs [role="tab"]{color:var(--muted) !important;background:var(--panel) !important;border:1px solid var(--line) !important;border-radius:8px !important;padding:.55rem 1.1rem !important;font-weight:700;height:auto !important}
.stTabs [role="tab"] p{color:inherit !important;font-weight:700 !important}
.stTabs [role="tab"]:hover{color:var(--ink) !important;border-color:var(--line2) !important}
.stTabs [role="tab"][aria-selected="true"]{color:#fff !important;background:linear-gradient(180deg,#22527f,#173a5c) !important;border-color:var(--blue2) !important;box-shadow:inset 0 -2px 0 var(--gold)}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none !important}
[data-testid="stTabContent"]{padding-top:1.3rem}
.ba-step{display:inline-flex;align-items:center;gap:.6rem;color:var(--steel2);font:700 .7rem "JetBrains Mono",monospace;letter-spacing:.16em;margin:0 0 .8rem}
.ba-step b{color:var(--gold);border:1px solid #d6a93655;border-radius:4px;padding:.1rem .4rem}

/* buttons */
.stButton > button,.stDownloadButton > button{min-height:2.75rem;border-radius:8px !important;font-weight:800;letter-spacing:.01em;transition:all .15s}
.stButton > button[kind="primary"],.stDownloadButton > button[kind="primary"]{color:#fff !important;border:1px solid #5d9bd6 !important;
  background:linear-gradient(180deg,#3479b8,#1f5a92) !important;box-shadow:0 8px 20px #1f5a9255,inset 0 1px 0 #ffffff33}
.stButton > button[kind="primary"]:hover{filter:brightness(1.12);transform:translateY(-1px)}
.stButton > button:not([kind="primary"]),.stDownloadButton > button:not([kind="primary"]){color:var(--ink) !important;background:var(--panel2) !important;border:1px solid var(--line2) !important}
.stButton > button:not([kind="primary"]):hover,.stDownloadButton > button:not([kind="primary"]):hover{border-color:var(--gold) !important;color:var(--gold2) !important}

/* inputs */
[data-baseweb="input"],[data-baseweb="select"] > div,[data-baseweb="textarea"],[data-testid="stNumberInputContainer"]{background:#0a0e15 !important;border:1px solid var(--line2) !important;border-radius:8px !important}
[data-baseweb="base-input"]{background:transparent !important}
[data-testid="stTextInputRootElement"]{background:#0a0e15 !important;border:1px solid var(--line2) !important;border-radius:8px !important}
[data-testid="stTextInputRootElement"]:focus-within{border-color:var(--blue2) !important;box-shadow:0 0 0 3px #2f6fa833}
[data-testid="stTextInputRootElement"] > div{background:transparent !important;border:0 !important}
[data-baseweb="input"]:focus-within,[data-baseweb="select"] > div:focus-within{border-color:var(--blue2) !important;box-shadow:0 0 0 3px #2f6fa833}
input,textarea,[data-baseweb="select"] span,[data-baseweb="select"] div{color:var(--ink) !important;-webkit-text-fill-color:var(--ink) !important}
[data-testid="stNumberInput"] button{background:#161d28 !important;color:var(--steel) !important;border-color:var(--line2) !important}
[data-baseweb="popover"] ul,[data-baseweb="menu"]{background:#121822 !important;border:1px solid var(--line2) !important}
[data-baseweb="popover"] li{color:var(--ink) !important} [data-baseweb="popover"] li:hover,[aria-selected="true"][role="option"]{background:#1d3550 !important}
[data-testid="stFileUploaderDropzone"]{background:#0e131b !important;border:1.5px dashed var(--line2) !important;border-radius:10px !important;padding:1.1rem}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--blue2) !important}
[data-testid="stFileUploaderDropzone"] span,[data-testid="stFileUploaderDropzone"] small{color:var(--muted) !important}
[data-testid="stFileUploaderDropzone"] button{background:var(--panel2) !important;color:var(--ink) !important;border:1px solid var(--line2) !important}
[data-baseweb="radio"] div,[data-baseweb="checkbox"] div{border-color:var(--steel2)}

/* cards, metrics, alerts */
[data-testid="stMetric"]{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:10px;padding:.95rem 1rem;position:relative;overflow:hidden}
[data-testid="stMetric"]:before{content:"";position:absolute;left:0;top:0;right:0;height:2px;background:linear-gradient(90deg,var(--gold),transparent 70%)}
[data-testid="stMetricLabel"] p{color:var(--muted) !important;font:600 .75rem "JetBrains Mono",monospace !important;letter-spacing:.08em}
[data-testid="stMetricValue"]{font:700 clamp(1.3rem,2vw,1.9rem) "JetBrains Mono",monospace !important;color:#fff !important}
[data-testid="stExpander"]{background:var(--panel);border:1px solid var(--line) !important;border-radius:10px}
[data-testid="stExpander"] summary{color:var(--ink) !important;font-weight:700}
[data-testid="stExpander"] summary:hover{color:var(--gold2) !important}
[data-testid="stAlert"]{border-radius:10px;border:1px solid var(--line2);background:#121a26 !important}
[data-testid="stImage"] img{border-radius:10px;border:1px solid var(--line2)}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:10px;overflow:hidden}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > .element-container .ba-panel-flag){background:var(--panel);border-color:var(--line) !important;border-radius:12px}
.ba-note{display:flex;gap:.6rem;align-items:flex-start;color:var(--muted);font-size:.8rem;line-height:1.6;border:1px solid var(--line);border-radius:10px;padding:.7rem .8rem;background:#0d1219}
.ba-note b{color:var(--ok)}

@media(max-width:820px){
  .ba-hero{grid-template-columns:1fr;gap:1.2rem}
  .ba-emblem{justify-self:center;max-width:320px;order:-1}
  .block-container{padding:1rem .9rem 3rem}
  .stTabs [data-baseweb="tab"]{padding:.5rem .65rem;font-size:.8rem}
}
</style>
"""


@lru_cache(maxsize=1)
def emblem_data_uri() -> str | None:
    if not EMBLEM.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(EMBLEM.read_bytes()).decode()


def page_icon():
    """Browser tab icon: the emblem file itself when present."""
    return str(EMBLEM) if EMBLEM.exists() else "🎯"


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def brand_header(tagline: str = "AI MARKSMANSHIP COACH") -> None:
    uri = emblem_data_uri()
    mark = f'<img src="{uri}" alt="SSAT 마크">' if uri else '<span class="ba-mark" aria-hidden="true"></span>'
    st.markdown(
        f'<div class="ba-brand">{mark}<span><span class="ba-name">BULLS<b>AI</b></span>'
        f'<span class="ba-tag">{escape(tagline)}</span></span></div>',
        unsafe_allow_html=True,
    )


def sidebar_nav() -> None:
    st.page_link("streamlit_app.py", label="사격 분석", icon=":material/my_location:")
    st.page_link("pages/1_라벨링.py", label="탄공 라벨링", icon=":material/label:")


def section(label: str) -> None:
    st.markdown(f'<div class="ba-sec">{escape(label)}</div>', unsafe_allow_html=True)


def step(num: str, label: str) -> None:
    st.markdown(f'<div class="ba-step"><b>{escape(num)}</b>{escape(label)}</div>', unsafe_allow_html=True)


def local_note() -> None:
    st.markdown('<div class="ba-note">🔒<span><b>LOCAL ONLY</b> — 모든 분석은 이 PC 안에서 처리됩니다. '
                '사진·영상은 외부로 전송되지 않습니다.</span></div>', unsafe_allow_html=True)


def hero(title: str, description: str, eyebrow: str = "SSAT · TARGET ANALYSIS",
         chips: list[tuple[str, str]] | None = None, show_emblem: bool = True) -> None:
    uri = emblem_data_uri() if show_emblem else None
    chips_html = "".join(f'<span class="ba-chip">{escape(k)} <b>{escape(v)}</b></span>' for k, v in (chips or []))
    fig = (f'<figure class="ba-emblem"><img src="{uri}" alt="SSAT 특공대 마크">'
           f'</figure>') if uri else ""
    st.markdown(
        f'<section class="ba-hero"><div><span class="ba-eyebrow">{escape(eyebrow)}</span>'
        f'<h1>{escape(title)}</h1><p>{escape(description)}</p>'
        f'<div class="ba-chips">{chips_html}</div></div>{fig}</section>',
        unsafe_allow_html=True,
    )
