"""로컬 탄공 라벨링 — 실제 해경 표적지 사진으로 파인튜닝 데이터 만들기 (이 PC 안에서만)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shootcoach.config import load_target_spec  # noqa: E402
from shootcoach.labeling import dataset_counts, draw_label_view, list_images, load_sample, prelabel, save_sample  # noqa: E402
from shootcoach.target.detect import Hole, get_detector  # noqa: E402
from shootcoach.target.markers import MarkerError  # noqa: E402

from app.theme import apply_theme, brand_header, page_icon, page_title, section, sidebar_nav  # noqa: E402

st.set_page_config(page_title="BullsAI · 탄공 라벨링", page_icon=page_icon(), layout="wide")
apply_theme()
page_title("탄공 라벨링")

configs = sorted((ROOT / "configs").glob("*.yaml"))
with st.sidebar:
    brand_header()
    sidebar_nav()
    section("DATASET")
    cfg = st.selectbox("표적지 설정", configs, format_func=lambda p: p.stem)
    src = st.text_input("원본 사진 폴더", str(ROOT / "data" / "haegyeong" / "raw"))
    out = st.text_input("데이터셋 저장 폴더", str(ROOT / "data" / "haegyeong" / "dataset"))
    rectified = st.checkbox("이미 정면 보정된 사진 (마커 없음)", False)
    counts = dataset_counts(out)
    st.caption(f"학습 {counts['train']['images']}장 · 탄공 {counts['train']['holes']}개 / 검증 {counts['val']['images']}장 · 탄공 {counts['val']['holes']}개")

DATA_ROOT = (ROOT / "data").resolve()


def _inside_data(path_str: str) -> Path | None:
    """Only folders under <repo>/data are allowed (the app may be reachable from the network)."""
    p = Path(path_str).expanduser()
    p = (p if p.is_absolute() else ROOT / p).resolve()
    return p if p == DATA_ROOT or DATA_ROOT in p.parents else None


src_p, out_p = _inside_data(src), _inside_data(out)
if src_p is None or out_p is None:
    st.error(f"보안상 폴더는 `{DATA_ROOT}` 아래만 쓸 수 있습니다.")
    st.stop()
src, out = str(src_p), str(out_p)
SPEC = load_target_spec(cfg)
imgs = list_images(src)
if not imgs:
    st.info(f"`{src}` 폴더에 사진(jpg/png)을 넣으세요.")
    st.stop()


@st.cache_resource
def detector():
    return get_detector("auto")


idx = min(max(st.session_state.get("idx", 0), 0), len(imgs) - 1)
path = imgs[idx]
key = f"{cfg.stem}:{path.name}"
if st.session_state.get("key") != key:
    try:
        rect, holes = prelabel(cv2.imread(str(path)), SPEC, detector(), rectified)
    except MarkerError as e:
        st.error(f"{path.name}: {e}")
        st.stop()
    saved = load_sample(out, path.stem, SPEC)
    st.session_state.update(key=key, rect=rect, holes=saved if saved is not None else holes, click=None,
                            loaded_saved=saved is not None)

rect, holes = st.session_state["rect"], st.session_state["holes"]

nav1, nav2, nav3 = st.columns([1, 6, 1], vertical_alignment="center")
if nav1.button("◀ 이전", width="stretch", disabled=idx == 0):
    st.session_state["idx"] = idx - 1
    st.rerun()
nav2.markdown(f"**{idx + 1} / {len(imgs)}** · {path.name} · 탄공 **{len(holes)}개**"
              + (" · 저장된 라벨" if st.session_state.get("loaded_saved") else ""))
if nav3.button("다음 ▶", width="stretch", disabled=idx >= len(imgs) - 1):
    st.session_state["idx"] = idx + 1
    st.rerun()

c1, c2 = st.columns([7, 3], gap="large")
with c1:
    view = draw_label_view(rect, holes, SPEC)
    send_w = min(1000, view.shape[1])
    small = cv2.resize(view, (send_w, int(view.shape[0] * send_w / view.shape[1])))
    click = streamlit_image_coordinates(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), width="stretch", key=f"img:{key}")
    if click and click != st.session_state.get("click"):
        st.session_state["click"] = click
        shown_w = click.get("width") or send_w           # displayed width in the browser
        k = view.shape[1] / shown_w / SPEC.px_per_mm      # displayed px → mm
        holes.append(Hole(click["x"] * k, click["y"] * k, SPEC.bullet_diameter_mm / 2, conf=1.0))
        st.rerun()
with c2:
    st.caption("사진 클릭 = 탄공 추가 · 주황 = 모델, 초록 = 추가")
    drop = st.multiselect("지울 번호", list(range(1, len(holes) + 1)), placeholder="오검출 번호 선택")
    b1, b2 = st.columns(2)
    if b1.button("삭제", width="stretch") and drop:
        st.session_state["holes"] = [h for i, h in enumerate(holes, 1) if i not in set(drop)]
        st.rerun()
    if b2.button("추가 취소", width="stretch") and holes:
        holes.pop()
        st.rerun()
    if st.button("저장 후 다음", type="primary", width="stretch"):
        p = save_sample(rect, holes, SPEC, out, path.stem)
        st.toast(f"저장: {p.name}")
        st.session_state["idx"] = min(len(imgs) - 1, idx + 1)
        st.rerun()
    with st.expander("학습 명령"):
        st.code(f"python scripts/train_detector.py --data {Path(out) / 'data.yaml'} --model models/hole_detector.pt --epochs 20 --name haegyeong",
                language="bash")
