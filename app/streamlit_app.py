"""BullsAI — AI 사격 교정 MVP (로컬 웹앱).

    streamlit run app/streamlit_app.py
같은 와이파이의 휴대폰에서 http://<이 PC IP>:8501 로 접속해 사진을 올릴 수 있다.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shootcoach.config import load_target_spec  # noqa: E402
from shootcoach.diagnosis.calibration import ShooterProfile, apply_profile  # noqa: E402
from shootcoach.diagnosis.rules import load_causes  # noqa: E402
from shootcoach.diagnosis.stage1 import diagnose_stage1  # noqa: E402
from shootcoach.target.scoring import group_stats  # noqa: E402
from shootcoach.diagnosis.stage1 import Candidate, Stage1Result, ZeroAdjust  # noqa: E402
from shootcoach.diagnosis.stage2 import diagnose_stage2  # noqa: E402
from shootcoach.explain.vlm import template_explanation, vlm_available, vlm_explanation  # noqa: E402
from shootcoach.pipeline import analyze_posture, analyze_target, to_json  # noqa: E402
from shootcoach.pose.features import arm_series  # noqa: E402
from shootcoach.pose.render import key_frames  # noqa: E402
from shootcoach.pose.simulate import Faults, simulate_side_view  # noqa: E402
from shootcoach.target.detect import ClassicHoleDetector, get_detector  # noqa: E402
from shootcoach.target.markers import MarkerError  # noqa: E402
from shootcoach.target.template import save_printable  # noqa: E402

from app.theme import apply_theme, brand_header, page_icon, section, sidebar_nav  # noqa: E402

st.set_page_config(page_title="BullsAI · AI 사격 교정", page_icon=page_icon(), layout="wide")
apply_theme()
SPEC = load_target_spec()


@st.cache_resource
def detector(kind: str):
    return ClassicHoleDetector() if kind == "classic" else get_detector("auto")


def stage1_from_dict(d: dict) -> Stage1Result:
    return Stage1Result(d["shape"], d["shape_ko"], d["sector"], d["sector_ko"], d["handedness"],
                        [Candidate(**c) for c in d["candidates"]],
                        ZeroAdjust(**d["zero_adjust"]) if d["zero_adjust"] else None, d["notes"])


with st.sidebar:
    brand_header()
    sidebar_nav()
    section("SHOOTER")
    hand = st.radio("주로 쓰는 손", ["right", "left"], format_func=lambda x: "오른손" if x == "right" else "왼손")
    section("RANGE")
    distance = st.number_input("사격 거리 (m)", 3.0, 50.0, 10.0, 1.0)
    section("ENGINE")
    det_kind = st.selectbox("탄공 검출기", ["auto", "classic"],
                            format_func=lambda x: "딥러닝 (YOLO)" if x == "auto" else "전통 영상처리 (대체)")
    shooter = st.text_input("사수 프로필", "")
    use_vlm = st.checkbox("AI 설명 문장", value=False)
    if use_vlm:
        st.caption("연결됨" if vlm_available() else "Ollama 꺼짐 — 기본 문장 사용")

tab1, tab2, tab3, tab4 = st.tabs(["표적지 분석", "자세 영상", "설명 문장", "표적지 인쇄"])

with tab1:
    demos = sorted(p.name for p in (ROOT / "samples").glob("demo_*.jpg"))
    i1, i2, i3 = st.columns([5, 4, 1.4], vertical_alignment="bottom")
    up = i1.file_uploader("표적지 사진", type=["jpg", "jpeg", "png"])
    demo = i2.selectbox("또는 데모 사진", ["(선택 안 함)"] + demos)
    go = i3.button("분석", type="primary", width="stretch")
    if go:
        img = None
        if up is not None:
            img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        elif demo != "(선택 안 함)":
            img = cv2.imread(str(ROOT / "samples" / demo))
        if img is None:
            st.warning("사진을 올리거나 데모 사진을 고르세요.")
        else:
            try:
                res = analyze_target(img, SPEC, hand, distance, None, detector(det_kind))
                if shooter.strip():
                    prof_path = ROOT / "profiles" / f"{shooter.strip()}.json"
                    if prof_path.exists():
                        st_ = group_stats(res.holes, SPEC)
                        s1 = apply_profile(diagnose_stage1(st_, SPEC, hand, distance, None), st_,
                                           ShooterProfile.load(prof_path))
                        res.report["stage1"] = s1.as_dict()
                st.session_state["report"] = res.report
                st.session_state["holes"] = res.holes
                st.session_state["overlay"] = res.overlay
                st.session_state.pop("stage2_seq", None)
            except MarkerError as e:
                st.error(str(e))
    rep = st.session_state.get("report")
    if not rep:
        st.caption("표적지 사진을 올리거나 데모 사진을 고른 뒤 분석을 누르세요.")
    else:
        g, s1 = rep["group"], rep["stage1"]
        left, right = st.columns([5, 6], gap="large")
        with left:
            st.image(cv2.cvtColor(st.session_state["overlay"], cv2.COLOR_BGR2RGB), width="stretch")
            st.download_button("리포트 JSON", to_json(rep), "report.json", "application/json", width="stretch")
        with right:
            m = st.columns(4)
            m[0].metric("탄 수", g["n"])
            m[1].metric("총점", g["total_score"])
            m[2].metric("오프셋 mm", f"{g['offset_mm']:.0f}")
            m[3].metric("반경 mm", f"{g['mean_radius_mm']:.0f}")
            st.subheader(f"{s1['shape_ko']} · {s1['sector_ko']}")
            if s1["zero_adjust"]:
                st.success(s1["zero_adjust"]["text_ko"])
            for n in s1["notes"]:
                st.info(n)
            for i, c in enumerate(s1["candidates"], 1):
                with st.expander(f"{i}. {c['cause_ko']}  ·  {c['score']:.2f}", expanded=i == 1):
                    st.write(c["guidance_ko"])
                    st.write(f"**교정 훈련:** {c['drill_ko']}")
                    st.caption(f"근거: {', '.join(c['sources'])} · 규칙 {', '.join(c['rule_ids'])}"
                               + (f" · 자세 영상: {c['observable_note']}" if c["observable_note"] else ""))
            if shooter.strip():
                with st.expander("진단 세션으로 기록 (캘리브레이션)"):
                    causes = load_causes()
                    cid = st.selectbox("일부러 낸 오류", sorted(causes), format_func=lambda k: causes[k].cause_ko)
                    if st.button("이 표적지를 기록"):
                        prof_path = ROOT / "profiles" / f"{shooter.strip()}.json"
                        prof = ShooterProfile.load(prof_path) if prof_path.exists() else ShooterProfile(shooter.strip(), hand)
                        prof.add_session(cid, group_stats(st.session_state["holes"], SPEC))
                        prof.save(prof_path)
                        st.success(f"기록했습니다 → profiles/{shooter.strip()}.json")
            st.caption(f"처리 {rep['timings']['total_s'] * 1000:.0f} ms")

with tab2:
    rep = st.session_state.get("report")
    if not rep:
        st.info("먼저 표적지 분석 탭에서 분석을 실행하세요.")
    else:
        i1, i2, i3 = st.columns([5, 4, 1.4], vertical_alignment="bottom")
        vid = i1.file_uploader("자세 영상", type=["mp4", "mov", "m4v"])
        sim = i2.selectbox("또는 데모 영상",
                           ["(선택 안 함)", "정상 자세", "격발 직전 총구 하강", "격발 직전 총구 들림 + 어깨 긴장",
                            "격발 직후 팔 내림", "조준 중 호흡 흔들림"])
        if i3.button("분석", type="primary", width="stretch", key="pose_go"):
            if vid is not None:
                with tempfile.NamedTemporaryFile(suffix=Path(vid.name).suffix, delete=False) as f:
                    f.write(vid.read())
                with st.spinner("관절 추출 중…"):
                    analyze_posture(rep, f.name, hand)
                st.session_state["stage2_video"] = f.name
            elif sim != "(선택 안 함)":
                faults = {"정상 자세": Faults(), "격발 직전 총구 하강": Faults(dip_deg=4),
                          "격발 직전 총구 들림 + 어깨 긴장": Faults(heel_deg=4, shrug=0.02),
                          "격발 직후 팔 내림": Faults(early_drop=0.12), "조준 중 호흡 흔들림": Faults(breath_amp=0.03)}[sim]
                seq = simulate_side_view((2.5, 5.0, 7.5), faults=faults)
                s2 = diagnose_stage2(stage1_from_dict(rep["stage1"]), seq, None, hand)
                rep["stage2"] = s2.as_dict() | {"shot_source": "motion (synthetic)"}
                st.session_state["stage2_seq"] = seq
        s2 = rep.get("stage2")
        if s2:
            seq = st.session_state.get("stage2_seq")
            left, right = st.columns([6, 5], gap="large")
            with left:
                st.subheader(s2["final_ko"])
                st.caption(f"{'측면' if s2['view'] == 'side' else '후방/정면'} 촬영 · 격발 {len(s2['shot_times'])}회")
                st.dataframe(pd.DataFrame([{"원인 후보": v["cause_ko"], "판정": v["status_ko"], "신뢰도": v["confidence"],
                                            "근거": " / ".join(v["evidence"])} for v in s2["verdicts"]]),
                             hide_index=True, width="stretch")
                for x in s2.get("extra_findings", []):
                    st.warning("추가 관찰: " + x)
                if seq is not None:
                    _, pitch, _ = arm_series(seq, hand)
                    st.line_chart(pd.DataFrame({"팔뚝 각도(°)": pitch}, index=np.round(seq.t, 2)), height=200, color="#cf3a30")
            with right:
                if seq is not None and s2["shot_times"]:
                    f = key_frames(seq, s2["shot_times"][0])
                    cols = st.columns(3)
                    for col, im, cap in zip(cols, f, ["0.3초 전", "격발 직전", "0.5초 후"]):
                        col.image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), caption=cap, width="stretch")
                for n in s2.get("notes", []):
                    st.caption(n)

with tab3:
    rep = st.session_state.get("report")
    if not rep:
        st.info("먼저 표적지 분석 탭에서 분석을 실행하세요.")
    elif st.button("설명 문장 만들기", type="primary"):
        if use_vlm:
            with st.spinner("로컬 VLM 생성 중…"):
                ex = vlm_explanation(rep, [st.session_state["overlay"]])
        else:
            ex = {"text": template_explanation(rep), "source": "template", "latency_s": 0}
        st.write(ex["text"])
        st.caption(f"출처: {'로컬 VLM ' + ex.get('model', '') if ex['source'] == 'vlm' else '템플릿'} · {ex['latency_s']}초"
                   + (f" · 오류: {ex['error']}" if ex.get("error") else ""))

with tab4:
    @st.cache_resource
    def printable():
        return save_printable(SPEC, Path(tempfile.gettempdir()) / "bullsai_target")   # 저장소 파일을 덮어쓰지 않음

    png, pdf = printable()
    c1, c2 = st.columns([1, 2])
    c1.image(str(png), width="stretch")
    c2.caption("A4 · 배율 100%로 인쇄")
    c2.download_button("PDF 받기", pdf.read_bytes(), pdf.name, "application/pdf", type="primary")
