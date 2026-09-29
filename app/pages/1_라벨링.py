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

from app.theme import apply_theme, brand_header, hero  # noqa: E402

st.set_page_config(page_title="BullsAI · 탄공 라벨링", layout="wide")
apply_theme()
hero("탄공 라벨링", "모델이 먼저 찾은 탄공을 확인하고 클릭해서 수정하세요. 모든 작업은 이 PC에서 처리됩니다.", "DATASET / LOCAL LABELING")

configs = sorted((ROOT / "configs").glob("*.yaml"))
with st.sidebar:
    brand_header("LABEL STUDIO")
    cfg = st.selectbox("표적지 설정", configs, format_func=lambda p: p.stem)
    src = st.text_input("원본 사진 폴더", str(ROOT / "data" / "haegyeong" / "raw"))
    out = st.text_input("데이터셋 저장 폴더", str(ROOT / "data" / "haegyeong" / "dataset"))
    rectified = st.checkbox("이미 정면 보정된 사진 (마커 없음)", False)
    display_w = st.slider("화면 표시 폭(px)", 400, 1000, 640, 20)
    st.write(dataset_counts(out))

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


idx = st.number_input(f"사진 번호 (1–{len(imgs)})", 1, len(imgs), st.session_state.get("idx", 1)) - 1
path = imgs[idx]
key = f"{cfg.stem}:{path.name}"
if st.session_state.get("key") != key:
    try:
        rect, holes = prelabel(cv2.imread(str(path)), SPEC, detector(), rectified)
    except MarkerError as e:
        st.error(f"{path.name}: {e}")
        st.stop()
    saved = load_sample(out, path.stem, SPEC)
    st.session_state.update(key=key, rect=rect, holes=saved if saved is not None else holes, click=None)
    if saved is not None:
        st.info("이미 저장된 라벨을 불러왔습니다.")

rect, holes = st.session_state["rect"], st.session_state["holes"]
c1, c2 = st.columns([3, 2])
with c1:
    view = draw_label_view(rect, holes, SPEC)
    scale = display_w / view.shape[1]
    small = cv2.resize(view, (display_w, int(view.shape[0] * scale)))
    click = streamlit_image_coordinates(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), key=f"img:{key}")
    if click and click != st.session_state.get("click"):
        st.session_state["click"] = click
        x_mm = click["x"] / scale / SPEC.px_per_mm
        y_mm = click["y"] / scale / SPEC.px_per_mm
        holes.append(Hole(x_mm, y_mm, SPEC.bullet_diameter_mm / 2, conf=1.0))
        st.rerun()
with c2:
    st.subheader(f"{path.name} · 탄공 {len(holes)}개")
    st.caption("모델이 제안한 탄공과 사람이 추가한 탄공이 함께 표시됩니다. 이미지를 클릭하면 탄공이 추가됩니다.")
    drop = st.multiselect("지울 번호 (오검출)", list(range(1, len(holes) + 1)))
    if st.button("선택 삭제") and drop:
        st.session_state["holes"] = [h for i, h in enumerate(holes, 1) if i not in set(drop)]
        st.rerun()
    if st.button("마지막 추가 취소") and holes:
        holes.pop()
        st.rerun()
    if st.button("저장 후 다음", type="primary"):
        p = save_sample(rect, holes, SPEC, out, path.stem)
        st.success(f"저장: {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
        st.session_state["idx"] = min(len(imgs), idx + 2)
        st.rerun()
    st.divider()
    st.code(f"python scripts/train_detector.py --data {Path(out) / 'data.yaml'} --model models/hole_detector.pt --epochs 20 --name haegyeong",
            language="bash")
    st.caption("파인튜닝은 이 PC에서 실행합니다. 50장 기준 수십 분.")
