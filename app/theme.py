"""Shared BullsAI bullseye and pixel arcade theme."""
from __future__ import annotations

from html import escape

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Black+Ops+One&family=IBM+Plex+Mono:wght@400;600;700&family=Noto+Sans+KR:wght@400;500;700;800&display=swap');
:root { --red:#ff343e; --ink:#f5f0e9; --muted:#aca2a3; --line:#49383b; --panel:#191719; }
html,body,.stApp,button,input,textarea,select { font-family:"Noto Sans KR",sans-serif !important; }
.stApp { color:var(--ink); background-color:#0c0c0e; background-image:linear-gradient(#ffffff04 1px,transparent 1px),linear-gradient(90deg,#ffffff04 1px,transparent 1px); background-size:24px 24px; }
.block-container { max-width:1280px; padding-top:2.4rem; padding-bottom:5rem; }
h1,h2,h3,h4,p,label,li,.stMarkdown,.stText,[data-testid="stMetricValue"] { color:var(--ink) !important; }
h1,h2,h3 { letter-spacing:-.035em; font-weight:800 !important; }
h1 { font-size:clamp(2rem,4vw,3.2rem) !important; }
.stCaption,[data-testid="stCaptionContainer"],small { color:var(--muted) !important; }
a { color:#ff7479 !important; } hr { border-color:var(--line) !important; }
[data-testid="stSidebar"] { background:#111113; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top:1.4rem; }
[data-testid="stSidebar"] .stMarkdown h6 { color:var(--red) !important; font:700 .73rem "IBM Plex Mono",monospace !important; letter-spacing:.12em; margin-top:2rem; }
.stTabs [data-baseweb="tab-list"] { gap:.35rem; border-bottom:1px solid var(--line); padding-bottom:.55rem; flex-wrap:wrap; }
.stTabs [data-baseweb="tab"] { color:var(--muted); background:var(--panel); border:1px solid var(--line); border-radius:0; padding:.65rem 1rem; font-weight:700; }
.stTabs [data-baseweb="tab"]:hover { color:var(--ink); border-color:var(--red); }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color:#fff !important; background:#801c27; border-color:var(--red); }
.stTabs [data-baseweb="tab-highlight"] { display:none; }
[data-testid="stTabContent"] { padding-top:1.4rem; }
.stButton > button,.stDownloadButton > button { min-height:2.7rem; border-radius:0 !important; font-weight:800; transition:background .15s,border-color .15s,transform .15s; }
.stButton > button:hover,.stDownloadButton > button:hover { transform:translateY(-2px); }
.stButton > button[kind="primary"],.stDownloadButton > button[kind="primary"] { color:#fff !important; background:var(--red) !important; border:1px solid #ff777c !important; box-shadow:4px 4px 0 #8b1a24; }
.stButton > button[kind="primary"]:hover,.stDownloadButton > button[kind="primary"]:hover { background:#db2631 !important; }
.stButton > button:not([kind="primary"]),.stDownloadButton > button:not([kind="primary"]) { color:var(--ink) !important; background:#231e20 !important; border:1px solid #695052 !important; }
.stButton > button:not([kind="primary"]):hover,.stDownloadButton > button:not([kind="primary"]):hover { border-color:var(--red) !important; }
[data-baseweb="input"],[data-baseweb="select"] > div,[data-baseweb="textarea"],[data-baseweb="base-input"],[data-testid="stFileUploaderDropzone"] { background:#1c1b1e !important; border-color:#59494c !important; border-radius:0 !important; color:var(--ink) !important; }
input,textarea,[data-baseweb="select"] span { color:var(--ink) !important; }
[data-testid="stFileUploaderDropzone"] { border-style:dashed !important; padding:1.2rem; }
[data-testid="stMetric"],[data-testid="stExpander"] { background:var(--panel); border:1px solid var(--line); border-radius:0; box-shadow:none; }
[data-testid="stMetric"] { padding:1rem; border-top:3px solid var(--red); }
[data-testid="stMetricLabel"] { color:var(--muted) !important; }
[data-testid="stMetricValue"] { font:700 clamp(1.3rem,2vw,2rem) "IBM Plex Mono",monospace !important; }
[data-testid="stExpander"] summary { color:var(--ink) !important; }
[data-testid="stAlert"],[data-testid="stDataFrame"],[data-testid="stImage"] img { border-radius:0; }
.ba-brand { display:flex; align-items:center; gap:.7rem; margin:.15rem 0 1rem; }
.ba-target { width:36px; height:36px; flex:none; border:2px solid var(--red); border-radius:50%; background:radial-gradient(circle,var(--red) 0 16%,#191518 17% 35%,var(--red) 36% 52%,#191518 53% 69%,var(--red) 70%); box-shadow:3px 3px 0 #7c1d25; }
.ba-name { color:var(--ink); font:400 1.6rem "Black Ops One","IBM Plex Mono",monospace; letter-spacing:.02em; }
.ba-name b { color:var(--red); font-weight:inherit; }
.ba-tag { display:block; color:var(--muted); font:600 .68rem "IBM Plex Mono",monospace; letter-spacing:.08em; }
.ba-hero { position:relative; overflow:hidden; min-height:265px; border:1px solid #713238; border-left:5px solid var(--red); background:radial-gradient(circle at 77% 55%,#66151e 0,#2b1218 21%,transparent 53%),linear-gradient(110deg,#23191b,#101012 68%,#290e15); padding:clamp(1.5rem,3.5vw,3rem); margin-bottom:1.7rem; box-shadow:0 18px 45px #0009,7px 7px 0 #260f14,inset 0 1px #a5535566; }
.ba-hero:before { content:""; position:absolute; inset:0; background:repeating-linear-gradient(0deg,#0000 0 3px,#0004 4px 5px),linear-gradient(135deg,#ffffff0a,transparent 35%); opacity:.5; pointer-events:none; }
.ba-hero:after { content:""; position:absolute; width:295px; height:295px; right:6%; top:-20px; border:1px solid #ff434a99; border-radius:50%; background:linear-gradient(90deg,transparent 49.8%,#ff52545c 50%,transparent 50.2%),linear-gradient(transparent 49.8%,#ff52545c 50%,transparent 50.2%),radial-gradient(circle,#ff5158 0 3%,#ff666b88 3.5% 4%,transparent 4.5% 17%,#ff494c99 17.4% 18%,transparent 18.5% 35%,#ff414677 35.4% 36%,transparent 36.5% 49%,#f7383d77 49.4% 50%,transparent 50.5%),radial-gradient(circle,#ff2c3466,transparent 55%); box-shadow:0 0 65px #d51e2966,inset 0 0 30px #e12d3244; filter:drop-shadow(0 0 13px #e52b39); pointer-events:none; }
.ba-eyebrow { color:#ff6a71; font:700 .76rem "IBM Plex Mono",monospace; letter-spacing:.17em; }
.ba-hero h1 { margin:.8rem 0 .4rem; position:relative; z-index:1; }
.ba-hero p { max-width:550px; color:#c8bfc0 !important; line-height:1.75; margin:0; position:relative; z-index:1; }
.ba-step { display:inline-block; color:#f7b2b5; font:600 .7rem "IBM Plex Mono",monospace; letter-spacing:.11em; margin:0 0 .6rem; }
@media(max-width:640px) { .block-container { padding:1.2rem .9rem 3rem; } .ba-hero { min-height:250px; } .ba-hero:after { right:-160px; opacity:.3; } .stTabs [data-baseweb="tab"] { padding:.5rem .6rem; font-size:.78rem; } }
</style>
"""


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def brand_header(tagline: str = "") -> None:
    tag = f'<span class="ba-tag">{escape(tagline)}</span>' if tagline else ""
    st.markdown(
        f'<div class="ba-brand"><span class="ba-target" aria-hidden="true"></span>'
        f'<span><span class="ba-name">BULLS<b>AI</b></span>{tag}</span></div>',
        unsafe_allow_html=True,
    )


def hero(title: str, description: str, eyebrow: str = "TARGET ACQUIRED // BULLSAI") -> None:
    st.markdown(
        f'<section class="ba-hero"><span class="ba-eyebrow">{escape(eyebrow)}</span>'
        f'<h1>{escape(title)}</h1><p>{escape(description)}</p></section>',
        unsafe_allow_html=True,
    )
